#!/usr/bin/env python
import sys, os, errno
from bup import options, git, vfs
from bup.helpers import *
try:
    import fuse
except ImportError:
    log('bup: error: The python "fuse" module is missing.\n' +
        'To use bup fuse, first install the python-fuse package.\n')
    sys.exit(1)


class Stat(fuse.Stat):
    def __init__(self):
        self.st_mode = 0
        self.st_ino = 0
        self.st_dev = 0
        self.st_nlink = 0
        self.st_uid = 0
        self.st_gid = 0
        self.st_size = 0
        self.st_atime = 0
        self.st_mtime = 0
        self.st_ctime = 0
        self.st_blocks = 0
        self.st_blksize = 0
        self.st_rdev = 0


cache = {}
def cache_get(top, path):
    parts = path.split('/')
    cache[('',)] = top
    c = None
    max = len(parts)
    #log('cache: %r\n' % cache.keys())
    for i in range(max):
        pre = parts[:max-i]
        #log('cache trying: %r\n' % pre)
        c = cache.get(tuple(pre))
        if c:
            rest = parts[max-i:]
            for r in rest:
                #log('resolving %r from %r\n' % (r, c.fullname()))
                c = c.lresolve(r)
                key = tuple(pre + [r])
                #log('saving: %r\n' % (key,))
                cache[key] = c
            break
    assert(c)
    return c
        
    
def _time_nsec_to_fuseStat(ns):
    # Actually, we want to do something like:
    #
    # return fuse.Timespec(tv_sec=int(int(ns) / 10**9), tv_nsec=int(ns) % 10**9)
    #
    # However, fuse-python (currently) does not support this, so we
    # convert it to float (and loose precision thereby!!).
    # Also, at the moment fuse-python ignores the fractional part and
    # the value must not be negative :)
    return ns / 1000000000.0 if ns > 0 else 0;


class BupFs(fuse.Fuse):
    def __init__(self, top, use_metadata=False):
        fuse.Fuse.__init__(self)
        self.top = top
        self._use_metadata = use_metadata
    
    def getattr(self, path):
        log('--getattr(%r)\n' % path)
        try:
            node = cache_get(self.top, path)
            st = Stat()
            st.st_nlink = node.nlinks(use_inode_cache=self._use_metadata)
            st.st_size = node.size()
            if self._use_metadata:
                st.st_mode = node.mode_meta_default()
                st.st_mtime = _time_nsec_to_fuseStat(node.mtime_nsec_meta_default())
                st.st_ctime = _time_nsec_to_fuseStat(node.ctime_nsec_meta_default())
                st.st_atime = _time_nsec_to_fuseStat(node.atime_nsec_meta_default())
                st.st_uid = node.uid_default(use_name=False)
                st.st_gid = node.gid_default(use_name=False)
                st.st_ino = node.inode
            else:
                st.st_mode = node.mode
                st.st_mtime = _time_nsec_to_fuseStat(node.mtime_nsec_default())
                st.st_ctime = _time_nsec_to_fuseStat(node.ctime_nsec_default())
                st.st_atime = _time_nsec_to_fuseStat(node.atime_nsec_default())
            return st
        except vfs.NoSuchFile:
            return -errno.ENOENT

    def readdir(self, path, offset):
        log('--readdir(%r)\n' % path)
        node = cache_get(self.top, path)
        if self._use_metadata:
            yield fuse.Direntry('.', ino=node.inode)
            yield fuse.Direntry('..', ino=node.inode_of_parent)
            for sub in node.subs():
                yield fuse.Direntry(sub.name, ino=sub.inode)
        else:
            yield fuse.Direntry('.')
            yield fuse.Direntry('..')
            for sub in node.subs():
                yield fuse.Direntry(sub.name)

    def readlink(self, path):
        log('--readlink(%r)\n' % path)
        node = cache_get(self.top, path)
        return node.readlink()

    def open(self, path, flags):
        log('--open(%r)\n' % path)
        node = cache_get(self.top, path)
        accmode = os.O_RDONLY | os.O_WRONLY | os.O_RDWR
        if (flags & accmode) != os.O_RDONLY:
            return -errno.EACCES
        node.open()

    def release(self, path, flags):
        log('--release(%r)\n' % path)

    def read(self, path, size, offset):
        log('--read(%r)\n' % path)
        n = cache_get(self.top, path)
        o = n.open()
        o.seek(offset)
        return o.read(size)


if not hasattr(fuse, '__version__'):
    raise RuntimeError, "your fuse module is too old for fuse.__version__"
fuse.fuse_python_api = (0, 2)


optspec = """
bup fuse [-d] [-f] <mountpoint>
--
d,debug   increase debug level
f,foreground  run in foreground
o,allow-other allow other users to access the filesystem
m,no-metadata do not load metadata (metadata is used by default)
"""
o = options.Options(optspec)
(opt, flags, extra) = o.parse(sys.argv[1:])

if len(extra) != 1:
    o.fatal("exactly one argument expected")

git.check_repo_or_die()
top = vfs.RefList(None)
f = BupFs(top, opt.metadata)
f.fuse_args.mountpoint = extra[0]
if opt.debug:
    f.fuse_args.add('debug')
if opt.metadata:
    f.fuse_args.add('use_ino')
if opt.foreground:
    f.fuse_args.setmod('foreground')
print f.multithreaded
f.multithreaded = False
if opt.allow_other:
    f.fuse_args.add('allow_other')

f.main()
