"""Virtual File System representing bup's repository contents.

The vfs.py library makes it possible to expose contents from bup's repository
and abstracts internal name mangling and storage from the exposition layer.
"""
import os, re, stat, time
from bup import git, metadata
from helpers import *
from bup.hashsplit import GIT_MODE_TREE, GIT_MODE_FILE

EMPTY_SHA='\0'*20

_cp = None
def cp():
    """Create a git.CatPipe object or reuse the already existing one."""
    global _cp
    if not _cp:
        _cp = git.CatPipe()
    return _cp

class NodeError(Exception):
    """VFS base exception."""
    pass

class NoSuchFile(NodeError):
    """Request of a file that does not exist."""
    pass

class NotDir(NodeError):
    """Attempt to do a directory action on a file that is not one."""
    pass

class NotFile(NodeError):
    """Access to a node that does not represent a file."""
    pass

class TooManySymlinks(NodeError):
    """Symlink dereferencing level is too deep."""
    pass


class InodeCache:
    """A class containing dictionaries to register and track the inodes for Nodes."""
    def __init__(self):
        # we start the idCount with 4 because to ids below are reserved:
        # 1 = parent mount directory
        # 2 = (first) RefList node
        # 3 = (first) CommitDir node
        # 4 = (first) TagDir node
        self._idCount = 4
        self._key_lookup = {}
        self._inode_lookup = { 1: 1 } # the inode #1 is reserved for the parent mount directory
    def _get_new_inode(self):
        while True:
            # search for a not yet used inode...
            self._idCount += 1
            if self._idCount not in self._inode_lookup:
                return self._idCount
    def get_inode_for_node(self, node):
        """
        Returns the inode for the given node. The inode is cached inside of
        node as node._inode. If no value is cached yet, a new one will
        be assigned and the node will be remembered.
        """
        inode = getattr(node, "_inode", None)
        if inode is not None:
            return inode
        hardlink_target = node.hardlink_target
        if hardlink_target is None:
            # nodes without hardlink_target get their individual inode
            inode = self._get_new_inode()
            self._inode_lookup[inode] = 1
            node._inode = inode
            return inode

        # do we want to share the inode between different snapshots???
        # Adjust the key accordingly.
        #
        #key = hardlink_target
        key = (node.commit_id(), hardlink_target)

        entry = self._key_lookup.get(key, None)
        if entry is None:
            # such a commit_id/hardlink_target pair is not yet seen. Create
            # a new inode.
            inode = self._get_new_inode()
            nlink = 1
            entry = (node, nlink)
        else:
            inode = None
            if isinstance(entry, tuple):
                # this is (by far) the normal case: the entry is a tuple.
                if node.same_content(entry[0]):
                    # as expected: we can share the inode because the nodes have
                    # the same content.
                    inode = entry[0]._inode
                    nlink = entry[1] + 1
                    entry = (entry[0], nlink)
                else:
                    # unexpected case: the hardlink_target for the nodes is the
                    # same, but their data/metadata is not. So, we change the
                    # entry to be a list of tuples. Below, a new inode gets assigned.
                    entry = [ entry ]
            else:
                # we search over the listed entries for this inode for a
                # matching node.
                for i,ientry in enumerate(entry):
                    node_i = ientry[0]
                    if node.same_content(node_i):
                        # the current entry has a matching node.
                        # increase the nlink count and use this inode.
                        inode = node_i._inode
                        nlink = ientry[1] + 1
                        entry[i] = (node_i, nlink)
                        break
            if inode is None:
                # The current node has the same hardlink_target as other nodes
                # but different data/metadata. This is actually an inconsistency
                # within the repository, which might happen, if the file system was
                # modified while making the backup.
                # In this case, we create a new inode for the node and append it to the entries list.
                inode = self._get_new_inode()
                nlink = 1
                entry.append( (node, nlink) )
        self._key_lookup[key] = entry
        self._inode_lookup[inode] = nlink
        node._inode = inode
        return inode
    def get_nlinks(self, node):
        """Return the link count for a give inode."""
        # The _inode_lookup dictionary will have an entry node.inode, because within the property
        # getter "inode" the value will be created (if it does not yet exist).
        return self._inode_lookup[node.inode]
    def try_set_inode_for_node(self, node, inode):
        """
        Try to set the inode to a certain value. If the node already has an
        inode or the inode is already registered, this method does nothing.
        The reason is that we don't want to use duplicate inodes, nor changing
        the inode of a node after setting it once.
        """
        if hasattr(node, "_inode") or inode in self._inode_lookup:
            return False
        self._inode_lookup[inode] = 1
        node._inode = inode
        return True
_inodeCache = InodeCache()


def _treeget(hash):
    it = cp().get(hash.encode('hex'))
    type = it.next()
    assert(type == 'tree')
    return git.tree_decode(''.join(it))


def _tree_decode(hash):
    tree = [(int(name,16),stat.S_ISDIR(mode),sha)
            for (mode,name,sha)
            in _treeget(hash)]
    assert(tree == list(sorted(tree)))
    return tree


def _chunk_len(hash):
    return sum(len(b) for b in cp().join(hash.encode('hex')))


def _last_chunk_info(hash):
    tree = _tree_decode(hash)
    assert(tree)
    (ofs,isdir,sha) = tree[-1]
    if isdir:
        (subofs, sublen) = _last_chunk_info(sha)
        return (ofs+subofs, sublen)
    else:
        return (ofs, _chunk_len(sha))


def _total_size(hash):
    (lastofs, lastsize) = _last_chunk_info(hash)
    return lastofs + lastsize


def _chunkiter(hash, startofs):
    assert(startofs >= 0)
    tree = _tree_decode(hash)

    # skip elements before startofs
    for i in xrange(len(tree)):
        if i+1 >= len(tree) or tree[i+1][0] > startofs:
            break
    first = i

    # iterate through what's left
    for i in xrange(first, len(tree)):
        (ofs,isdir,sha) = tree[i]
        skipmore = startofs-ofs
        if skipmore < 0:
            skipmore = 0
        if isdir:
            for b in _chunkiter(sha, skipmore):
                yield b
        else:
            yield ''.join(cp().join(sha.encode('hex')))[skipmore:]


class _ChunkReader:
    def __init__(self, hash, isdir, startofs):
        if isdir:
            self.it = _chunkiter(hash, startofs)
            self.blob = None
        else:
            self.it = None
            self.blob = ''.join(cp().join(hash.encode('hex')))[startofs:]
        self.ofs = startofs

    def next(self, size):
        out = ''
        while len(out) < size:
            if self.it and not self.blob:
                try:
                    self.blob = self.it.next()
                except StopIteration:
                    self.it = None
            if self.blob:
                want = size - len(out)
                out += self.blob[:want]
                self.blob = self.blob[want:]
            if not self.it:
                break
        debug2('next(%d) returned %d\n' % (size, len(out)))
        self.ofs += len(out)
        return out


class _FileReader(object):
    def __init__(self, hash, size, isdir):
        self.hash = hash
        self.ofs = 0
        self.size = size
        self.isdir = isdir
        self.reader = None

    def seek(self, ofs):
        if ofs > self.size:
            self.ofs = self.size
        elif ofs < 0:
            self.ofs = 0
        else:
            self.ofs = ofs

    def tell(self):
        return self.ofs

    def read(self, count = -1):
        if count < 0:
            count = self.size - self.ofs
        if not self.reader or self.reader.ofs != self.ofs:
            self.reader = _ChunkReader(self.hash, self.isdir, self.ofs)
        try:
            buf = self.reader.next(count)
        except:
            self.reader = None
            raise  # our offsets will be all screwed up otherwise
        self.ofs += len(buf)
        return buf

    def close(self):
        pass


class Node:
    """Base class for file representation."""
    def __init__(self, parent, name, mode, hash):
        self.parent = parent
        self.name = name
        self.mode = mode
        self.hash = hash
        self.atime_nsec = self.ctime_nsec = self.mtime_nsec = None
        self._subs = None


    def _get_atime_nsec_meta(self):
        """Return the atime in nano seconds from the metadata or None if it is missing"""
        metadata = self.metadata()
        if metadata is not None:
            return getattr(metadata, 'atime', None)
    atime_nsec_meta = property(_get_atime_nsec_meta)
    atime = property(lambda self: self.atime_nsec / 1000000000.0 if self.atime_nsec is not None else None)
    def atime_nsec_meta_default(self, default=0, use_atime_nsec=True):
        """Return the atime in nano seconds from the metadata. If the metadata is missing, take the value from atime_nsec (if use_atime_nsec) or return default."""
        time = self._get_atime_nsec_meta()
        if time is not None:
            return time
        return self.atime_nsec if use_atime_nsec and self.atime_nsec is not None else default
    def atime_nsec_default(self, default=0):
        """Return the nodes atime_nsec or a default if the value is missing (does not use metadata) """
        return self.atime_nsec if self.atime_nsec is not None else default

    def _get_ctime_nsec_meta(self):
        """Return the ctime in nano seconds from the metadata or None if it is missing"""
        metadata = self.metadata()
        if metadata is not None:
            return getattr(metadata, 'ctime', None)
    ctime_nsec_meta = property(_get_ctime_nsec_meta)
    ctime = property(lambda self: self.ctime_nsec / 1000000000.0 if self.ctime_nsec is not None else None)
    def ctime_nsec_meta_default(self, default=0, use_ctime_nsec=True):
        """Return the ctime in nano seconds from the metadata. If the metadata is missing, take the value from ctime_nsec (if use_ctime_nsec) or return default."""
        time = self._get_ctime_nsec_meta()
        if time is not None:
            return time
        return self.ctime_nsec if use_ctime_nsec and self.ctime_nsec is not None else default
    def ctime_nsec_default(self, default=0):
        """Return the nodes ctime_nsec or a default if the value is missing (does not use metadata) """
        return self.ctime_nsec if self.ctime_nsec is not None else default

    def _get_mtime_nsec_meta(self):
        """Return the mtime in nano seconds from the metadata or None if it is missing"""
        metadata = self.metadata()
        if metadata is not None:
            return getattr(metadata, 'mtime', None)
    mtime_nsec_meta = property(_get_mtime_nsec_meta)
    mtime = property(lambda self: self.mtime_nsec / 1000000000.0 if self.mtime_nsec is not None else None)
    def mtime_nsec_meta_default(self, default=0, use_mtime_nsec=True):
        """Return the mtime in nano seconds from the metadata. If the metadata is missing, take the value from mtime_nsec (if use_mtime_nsec) or return default."""
        time = self._get_mtime_nsec_meta()
        if time is not None:
            return time
        return self.mtime_nsec if use_mtime_nsec and self.mtime_nsec is not None else default
    def mtime_nsec_default(self, default=0):
        """Return the nodes mtime_nsec or a default if the value is missing (does not use metadata) """
        return self.mtime_nsec if self.mtime_nsec is not None else default

    def _set_time_nsec_from_git_date(self, date):
        """Set the [acm]time_nsec values to the given date (date is in seconds)"""
        self.atime_nsec = self.ctime_nsec = self.mtime_nsec = date * 1000000000


    def _get_mode_meta(self):
        """Returns the mode from the metadata, or None if the metadata is not available"""
        metadata = self.metadata()
        return getattr(metadata, 'mode', None) if metadata is not None else None
    mode_meta = property(_get_mode_meta)
    def mode_meta_default(self):
        """ Returns the mode from the metadata (self.mode_meta) and as fallback the static mode of the node (self.meta) """
        metadata = self.metadata()
        if metadata is not None:
            mode = getattr(metadata, 'mode', None)
            if mode is not None:
                return mode
        return self.mode


    def _get_uid(self):
        """Return the uid from metadata or None if the metadata is missing"""
        metadata = self.metadata()
        return getattr(metadata, 'uid', None) if metadata is not None else None
    uid = property(_get_uid)

    def _get_user(self):
        """Return the user from metadata or None if the metadata is missing"""
        metadata = self.metadata()
        return getattr(metadata, 'user', None) if metadata is not None else None
    user = property(_get_user);

    def uid_default(self, use_name=False):
        """
        Returns the uid of the node with several fallback options.

        If use_name is False, it returns the uid from the metadata or
        the user id of the current process (if the metadata is missing)

        If use_name is True, it takes metadata().user and tries to resolve the user id from
        the user name. If that fails, it acts as if use_name is False.
        """
        metadata = self.metadata()
        if metadata is not None:
            if use_name:
                name = getattr(metadata, 'user', None)
                if name:
                    entry = pwd_from_name(name)
                    if entry:
                        return entry.pw_uid
            uid = getattr(metadata, 'uid', None)
            if uid is not None:
                return uid
        return getuid()


    def _get_gid(self):
        """Returns the gid from the metadata or None if the metadata is missing"""
        metadata = self.metadata()
        return getattr(metadata, 'gid', None) if metadata is not None else None
    gid = property(_get_gid)

    def _get_group(self):
        """Returns the group name from the metadata or None if the metadata is missing"""
        metadata = self.metadata()
        return getattr(metadata, 'group', None) if metadata is not None else None
    group = property(_get_group);

    def gid_default(self, use_name=False):
        """
        Returns the gid of the node with several fallback options.

        If use_name is False, it returns the gid from the metadata or
        the group id of the current process (if the metadata is missing).

        If use_name is True, it takes metadata().group and tries to resolve the group id from
        the group name. If that fails, it acts as if use_name is False.
        """
        metadata = self.metadata()
        if metadata is not None:
            if use_name:
                name = getattr(metadata, 'group', None)
                if name:
                    entry = grp_from_name(name)
                    if entry:
                        return entry.gr_gid
            gid = getattr(metadata, 'gid', None)
            if gid is not None:
                return gid
        return getgid()


    def commit_id(self):
        """Returns the id of the commit to which the current node belongs, or None otherwise"""
        while self:
            if hasattr(self, '_commit_id'):
                return self._commit_id
            self = self.parent


    def __repr__(self):
        return "<%s object at X - name:%r hash:%s parent:%r>" \
            % (self.__class__, self.name, self.hash.encode('hex'),
               self.parent.name if self.parent else None)

    def __cmp__(a, b):
        if a is b:
            return 0
        return (cmp(a and a.parent, b and b.parent) or
                cmp(a and a.name, b and b.name))

    def same_content(self, other):
        """Returns true, if self and the other node have the same content, type and metadata."""
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        if self.mode != other.mode or self.hash != other.hash:
            return False
        a_meta = self.metadata()
        b_meta = other.metadata()
        return a_meta is b_meta or (a_meta is not None and b_meta is not None and a_meta.same_file(b_meta))

    def __iter__(self):
        return iter(self.subs())

    def fullname(self, stop_at=None):
        """Get this file's full path."""
        assert(self != stop_at)  # would be the empty string; too weird
        if self.parent and self.parent != stop_at:
            return os.path.join(self.parent.fullname(stop_at=stop_at),
                                self.name)
        else:
            return self.name

    def _mksubs(self):
        self._subs = {}

    def subs(self):
        """Get a list of nodes that are contained in this node."""
        if self._subs == None:
            self._mksubs()
        return sorted(self._subs.values())

    def sub(self, name):
        """Get node named 'name' that is contained in this node."""
        if self._subs == None:
            self._mksubs()
        ret = self._subs.get(name)
        if not ret:
            raise NoSuchFile("no file %r in %r" % (name, self.name))
        return ret

    def top(self):
        """Return the very top node of the tree."""
        if self.parent:
            return self.parent.top()
        else:
            return self

    def fs_top(self):
        """Return the top node of the particular backup set.

        If this node isn't inside a backup set, return the root level.
        """
        if self.parent and not isinstance(self.parent, CommitList):
            return self.parent.fs_top()
        else:
            return self

    def _lresolve(self, parts):
        #debug2('_lresolve %r in %r\n' % (parts, self.name))
        if not parts:
            return self
        (first, rest) = (parts[0], parts[1:])
        if first == '.':
            return self._lresolve(rest)
        elif first == '..':
            if not self.parent:
                raise NoSuchFile("no parent dir for %r" % self.name)
            return self.parent._lresolve(rest)
        elif rest:
            return self.sub(first)._lresolve(rest)
        else:
            return self.sub(first)

    def lresolve(self, path, stay_inside_fs=False):
        """Walk into a given sub-path of this node.

        If the last element is a symlink, leave it as a symlink, don't resolve
        it.  (like lstat())
        """
        start = self
        if not path:
            return start
        if path.startswith('/'):
            if stay_inside_fs:
                start = self.fs_top()
            else:
                start = self.top()
            path = path[1:]
        parts = re.split(r'/+', path or '.')
        if not parts[-1]:
            parts[-1] = '.'
        #debug2('parts: %r %r\n' % (path, parts))
        return start._lresolve(parts)

    def resolve(self, path = ''):
        """Like lresolve(), and dereference it if it was a symlink."""
        return self.lresolve(path).lresolve('.')

    def try_resolve(self, path = ''):
        """Like resolve(), but don't worry if a symlink uses an invalid path.

        Returns an error if any intermediate nodes were invalid.
        """
        n = self.lresolve(path)
        try:
            n = n.lresolve('.')
        except NoSuchFile:
            pass
        return n

    def _get_hardlink_target(self):
        metadata = self.metadata()
        return getattr(metadata, 'hardlink_target', None) if metadata is not None else None
    hardlink_target = property(_get_hardlink_target)

    inode = property(_inodeCache.get_inode_for_node)
    def _try_set_inode(self, inode):
        """
        Try to set the inode to a certain value. If the node already has an
        inode or the inode is already registred, this method does nothing.
        We don't want to use duplicate inodes, nor changing the inode of a node.
        """
        return _inodeCache.try_set_inode_for_node(self, inode)

    def _get_inode_of_parent(self):
        """
        Get the inode of the parent node or 1 if there is no parent. The 1 is on purpose, because it
        indicates the inode for the parent directory of the fuse mount point.
        """
        parent = self.parent
        return parent.inode if parent is not None else 1
    inode_of_parent = property(_get_inode_of_parent)

    def nlinks(self, use_inode_cache=False):
        """
        Get the number of hard links to the current node.

        For directories, this is the number of subdirectories in the directory + 2.

        For other nodes it is 1 (if use_inode_cache is False). When use_inode_cache is
        True, the inode is used to lookup the link count. This means, that the
        metadata will be accessed. Actually, the number returned in this case will
        only be correct, if every other instance of the file was already visited, because the
        nlinks are only counted, as we encounter the same file. The reason is that
        before we did not encounter all instances, we don't know that they even exist.
        """
        if self._subs == None:
            self._mksubs()
        if stat.S_ISDIR(self.mode):
            return 2 + sum( 1 for item in self if stat.S_ISDIR(item.mode) )

        if not use_inode_cache:
            return 1

        # This approach has the problem, that it will only report
        # the proper link count for all equal nodes that were already visited.
        # IOW, for this to be correct, you have to visit very node once
        # and then visit your node again.
        return _inodeCache.get_nlinks(self)

    def size(self):
        """Get the size of the current node."""
        return 0

    def open(self):
        """Open the current node. It is an error to open a non-file node."""
        raise NotFile('%s is not a regular file' % self.name)

    def _populate_metadata(self):
        # Only Dirs contain .bupm files, so by default, do nothing.
        self._metadata = None

    def metadata(self):
        """Return this Node's Metadata() object, if any."""
        if not hasattr(self, "_metadata"):
            if self.parent:
                self.parent._populate_metadata()
            if not hasattr(self, "_metadata"):
                self._metadata = None
        return self._metadata


class File(Node):
    """A normal file from bup's repository."""
    def __init__(self, parent, name, mode, hash, bupmode):
        Node.__init__(self, parent, name, mode, hash)
        self.bupmode = bupmode
        self._cached_size = None
        self._filereader = None

    def open(self):
        """Open the file."""
        # You'd think FUSE might call this only once each time a file is
        # opened, but no; it's really more of a refcount, and it's called
        # once per read().  Thus, it's important to cache the filereader
        # object here so we're not constantly re-seeking.
        if not self._filereader:
            self._filereader = _FileReader(self.hash, self.size(),
                                           self.bupmode == git.BUP_CHUNKED)
        self._filereader.seek(0)
        return self._filereader

    def size(self):
        """Get this file's size."""
        if self._cached_size == None:
            debug1('<<<<File.size() is calculating (for %r)...\n' % self.name)
            if self.bupmode == git.BUP_CHUNKED:
                self._cached_size = _total_size(self.hash)
            else:
                self._cached_size = _chunk_len(self.hash)
            debug1('<<<<File.size() done.\n')
        return self._cached_size


_symrefs = 0
class Symlink(File):
    """A symbolic link from bup's repository."""
    def __init__(self, parent, name, hash, bupmode):
        File.__init__(self, parent, name, 0120000, hash, bupmode)

    def size(self):
        """Get the file size of the file at which this link points."""
        return len(self.readlink())

    def readlink(self):
        """Get the path that this link points at."""
        return ''.join(cp().join(self.hash.encode('hex')))

    def dereference(self):
        """Get the node that this link points at.

        If the path is invalid, raise a NoSuchFile exception. If the level of
        indirection of symlinks is 100 levels deep, raise a TooManySymlinks
        exception.
        """
        global _symrefs
        if _symrefs > 100:
            raise TooManySymlinks('too many levels of symlinks: %r'
                                  % self.fullname())
        _symrefs += 1
        try:
            try:
                return self.parent.lresolve(self.readlink(),
                                            stay_inside_fs=True)
            except NoSuchFile:
                raise NoSuchFile("%s: broken symlink to %r"
                                 % (self.fullname(), self.readlink()))
        finally:
            _symrefs -= 1

    def _lresolve(self, parts):
        return self.dereference()._lresolve(parts)


class FakeSymlink(Symlink):
    """A symlink that is not stored in the bup repository."""
    def __init__(self, parent, name, toname):
        Symlink.__init__(self, parent, name, EMPTY_SHA, git.BUP_NORMAL)
        self.toname = toname

    def readlink(self):
        """Get the path that this link points at."""
        return self.toname


class Dir(Node):
    """A directory stored inside of bup's repository."""

    def __init__(self, *args):
        Node.__init__(self, *args)
        self._bupm = None

    def _populate_metadata(self):
        if not self._subs:
            self._mksubs()
        if not self._bupm:
            self._metadata = None
            return
        meta_stream = self._bupm.open()
        self._metadata = metadata.Metadata.read(meta_stream)
        for sub in self:
            if not stat.S_ISDIR(sub.mode):
                sub._metadata = metadata.Metadata.read(meta_stream)

    def _mksubs(self):
        self._subs = {}
        it = cp().get(self.hash.encode('hex'))
        type = it.next()
        if type == 'commit':
            del it
            it = cp().get(self.hash.encode('hex') + ':')
            type = it.next()
        assert(type == 'tree')
        for (mode,mangled_name,sha) in git.tree_decode(''.join(it)):
            if mangled_name == '.bupm':
                self._bupm = File(self, mangled_name, mode, sha, git.BUP_NORMAL)
                continue
            name = mangled_name
            (name,bupmode) = git.demangle_name(mangled_name)
            if bupmode == git.BUP_CHUNKED:
                mode = GIT_MODE_FILE
            if stat.S_ISDIR(mode):
                self._subs[name] = Dir(self, name, mode, sha)
            elif stat.S_ISLNK(mode):
                self._subs[name] = Symlink(self, name, sha, bupmode)
            else:
                self._subs[name] = File(self, name, mode, sha, bupmode)

    def metadata(self):
        """Return this Dir's Metadata() object, if any."""
        if not hasattr(self, "_metadata"):
            self._populate_metadata()
        return self._metadata

    def metadata_file(self):
        """Return this Dir's .bupm File, if any."""
        self._populate_metadata()
        return self._bupm


class CommitDir(Node):
    """A directory that contains all commits that are reachable by a ref.

    Contains a set of subdirectories named after the commits' first byte in
    hexadecimal. Each of those directories contain all commits with hashes that
    start the same as the directory name. The name used for those
    subdirectories is the hash of the commit without the first byte. This
    separation helps us avoid having too much directories on the same level as
    the number of commits grows big.
    """
    def __init__(self, parent, name):
        Node.__init__(self, parent, name, GIT_MODE_TREE, EMPTY_SHA)

    def _mksubs(self):
        self._subs = {}
        refs = git.list_refs()
        for ref in refs:
            #debug2('ref name: %s\n' % ref[0])
            revs = git.rev_list(ref[1].encode('hex'))
            for (date, commit) in revs:
                #debug2('commit: %s  date: %s\n' % (commit.encode('hex'), date))
                commithex = commit.encode('hex')
                containername = commithex[:2]
                dirname = commithex[2:]
                n1 = self._subs.get(containername)
                if not n1:
                    n1 = CommitList(self, containername)
                    self._subs[containername] = n1

                if n1.commits.get(dirname):
                    # Stop work for this ref, the rest should already be present
                    break

                n1.commits[dirname] = (commit, date)


class CommitList(Node):
    """A list of commits with hashes that start with the current node's name."""
    def __init__(self, parent, name):
        Node.__init__(self, parent, name, GIT_MODE_TREE, EMPTY_SHA)
        self.commits = {}

    def _mksubs(self):
        self._subs = {}
        for (name, (hash, date)) in self.commits.items():
            n1 = Dir(self, name, GIT_MODE_TREE, hash)
            n1._set_time_nsec_from_git_date(date)
            n1._commit_id = hash
            self._subs[name] = n1


class TagDir(Node):
    """A directory that contains all tags in the repository."""
    def __init__(self, parent, name):
        Node.__init__(self, parent, name, GIT_MODE_TREE, EMPTY_SHA)

    def _mksubs(self):
        self._subs = {}
        for (name, sha) in git.list_refs():
            if name.startswith('refs/tags/'):
                name = name[10:]
                date = git.rev_get_date(sha.encode('hex'))
                commithex = sha.encode('hex')
                target = '../.commit/%s/%s' % (commithex[:2], commithex[2:])
                tag1 = FakeSymlink(self, name, target)
                tag1._set_time_nsec_from_git_date(date)
                self._subs[name] = tag1


class BranchList(Node):
    """A list of links to commits reachable by a branch in bup's repository.

    Represents each commit as a symlink that points to the commit directory in
    /.commit/??/ . The symlink is named after the commit date.
    """
    def __init__(self, parent, name, hash):
        Node.__init__(self, parent, name, GIT_MODE_TREE, hash)

    def _mksubs(self):
        self._subs = {}

        tags = git.tags()

        revs = list(git.rev_list(self.hash.encode('hex')))
        latest = revs[0]
        for (date, commit) in revs:
            l = time.localtime(date)
            ls = time.strftime('%Y-%m-%d-%H%M%S', l)
            commithex = commit.encode('hex')
            target = '../.commit/%s/%s' % (commithex[:2], commithex[2:])
            n1 = FakeSymlink(self, ls, target)
            n1._set_time_nsec_from_git_date(date)
            self._subs[ls] = n1

            for tag in tags.get(commit, []):
                t1 = FakeSymlink(self, tag, target)
                t1._set_time_nsec_from_git_date(date)
                self._subs[tag] = t1

        (date, commit) = latest
        commithex = commit.encode('hex')
        target = '../.commit/%s/%s' % (commithex[:2], commithex[2:])
        n1 = FakeSymlink(self, 'latest', target)
        n1._set_time_nsec_from_git_date(date)
        self._subs['latest'] = n1


class RefList(Node):
    """A list of branches in bup's repository.

    The sub-nodes of the ref list are a series of CommitList for each commit
    hash pointed to by a branch.

    Also, a special sub-node named '.commit' contains all commit directories
    that are reachable via a ref (e.g. a branch).  See CommitDir for details.
    """
    def __init__(self, parent):
        Node.__init__(self, parent, '/', GIT_MODE_TREE, EMPTY_SHA)
        self._try_set_inode(2)

    def _mksubs(self):
        self._subs = {}

        commit_dir = CommitDir(self, '.commit')
        self._subs['.commit'] = commit_dir
        commit_dir._try_set_inode(3)

        tag_dir = TagDir(self, '.tag')
        self._subs['.tag'] = tag_dir
        tag_dir._try_set_inode(4)

        for (name,sha) in git.list_refs():
            if name.startswith('refs/heads/'):
                name = name[11:]
                date = git.rev_get_date(sha.encode('hex'))
                n1 = BranchList(self, name, sha)
                n1._set_time_nsec_from_git_date(date)
                self._subs[name] = n1
