//! Darwin-specific adapters for the shared POSIX shims.
//!
//! macOS and iOS/iPadOS use `funopen` for virtual stdio streams and share the
//! carrier-file-descriptor, directory, metadata, and path-routing behavior in the
//! parent module. The declarations below use the Darwin libc ABI directly.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use super::{
    CookieState, PANIC_ERRNO, cookie_close_state, cookie_read_into, cookie_seek_to,
    cookie_write_from, orig_fn, set_errno,
};
use crate::dispatch::{HookScope, Sym, current_engine, guard_abi};
use crate::router::Engine;
use core::ffi::{c_char, c_int, c_void};
use libc::{FILE, off_t, size_t, ssize_t};
use std::path::Path;

// `funopen`'s callbacks. macOS `fpos_t` is `off_t` (`i64`), ABI-identical here.
type Fpos = off_t;
type FunopenRead = unsafe extern "C" fn(*mut c_void, *mut c_char, c_int) -> c_int;
type FunopenWrite = unsafe extern "C" fn(*mut c_void, *const c_char, c_int) -> c_int;
type FunopenSeek = unsafe extern "C" fn(*mut c_void, Fpos, c_int) -> Fpos;
type FunopenClose = unsafe extern "C" fn(*mut c_void) -> c_int;

unsafe extern "C" {
    /// BSD `FILE *funopen(const void *cookie, readfn, writefn, seekfn, closefn)`.
    /// Returns a real `FILE*` driven by the supplied callbacks, or null on failure
    /// (in which case it does **not** take ownership of `cookie`).
    fn funopen(
        cookie: *const c_void,
        readfn: Option<FunopenRead>,
        writefn: Option<FunopenWrite>,
        seekfn: Option<FunopenSeek>,
        closefn: Option<FunopenClose>,
    ) -> *mut FILE;
}

/// `funopen` read callback A?€�t adapts the shared logic to the BSD `int`-sized ABI.
extern "C" fn funopen_read(cookie: *mut c_void, buf: *mut c_char, nbytes: c_int) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let len = usize::try_from(nbytes).unwrap_or(0);
        let read = cookie_read_into(cookie, buf.cast::<u8>(), len);
        c_int::try_from(read).unwrap_or(c_int::MAX)
    })
}

extern "C" fn funopen_write(cookie: *mut c_void, buf: *const c_char, nbytes: c_int) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let len = usize::try_from(nbytes).unwrap_or(0);
        let written = cookie_write_from(cookie, buf.cast::<u8>(), len);
        c_int::try_from(written).unwrap_or(c_int::MAX)
    })
}

/// `funopen` seek callback A?€�t returns the new absolute offset by value (`fpos_t`).
extern "C" fn funopen_seek(cookie: *mut c_void, offset: Fpos, whence: c_int) -> Fpos {
    guard_abi(-1, PANIC_ERRNO, || {
        cookie_seek_to(cookie, offset, whence).unwrap_or(-1)
    })
}

/// `funopen` close callback A?€�t reclaim the boxed state exactly once.
extern "C" fn funopen_close(cookie: *mut c_void) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        cookie_close_state(cookie);
        0
    })
}

/// Open a virtual path as a real `funopen`-backed `FILE*`. A read-only stream has a
/// **null write callback** so any `fwrite`/`fprintf` fails predictably (BSD
/// so `fwrite`/`fprintf` reach the VFS. Returns null with `errno` on failure
/// (mode/write/ENOENT/provider gate handled by [`super::prepare_cookie`]).
pub(super) fn fopen_virtual(engine: &Engine, path: &Path, mode: *const c_char) -> *mut FILE {
    let Some(cookie) = super::prepare_cookie(engine, path, mode) else {
        return core::ptr::null_mut();
    };
    // SAFETY: `cookie` is the live `CookieState` just boxed by `prepare_cookie`.
    let writable = unsafe { (*cookie).writable };
    // SAFETY: `cookie` is a live `CookieState`; on success `funopen` takes ownership
    // and reclaims it via `funopen_close`.
    let file = unsafe {
        funopen(
            cookie.cast::<c_void>(),
            Some(funopen_read),
            writable.then_some(funopen_write as FunopenWrite),
            Some(funopen_seek),
            Some(funopen_close),
        )
    };
    if file.is_null() {
        // funopen failed: it did not take ownership; reclaim the cookie.
        // SAFETY: `cookie` was not consumed by the failed `funopen`.
        drop(unsafe { Box::from_raw(cookie.cast::<CookieState>()) });
        set_errno(libc::EIO);
    }
    file
}

/// `__getdirentries64(int fd, void *buf, size_t bufsize, off_t *basep)` A?€�t the
/// low-level directory-read wrapper `readdir` bottoms out in. A virtual carrier fd is
/// a regular file (`/dev/null`), never a directory, so fail closed (`ENOTDIR`);
/// everything else passes through. (Our `readdir`/`opendir` are hooked, so a virtual
/// `DIR*` is served there and never reaches this path.)
pub(super) extern "C" fn shim_getdirentries64(
    fd: c_int,
    buf: *mut c_void,
    bufsize: size_t,
    basep: *mut off_t,
) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnGetdirentries64 = orig_fn(Sym::Getdirentries64);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, buf, bufsize, basep) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().get_file(fd as usize).is_some() {
            set_errno(libc::ENOTDIR); // a virtual carrier fd is not a directory
            -1
        } else {
            pass()
        }
    })
}

type PfnGetdirentries64 = unsafe extern "C" fn(c_int, *mut c_void, size_t, *mut off_t) -> ssize_t;

// ---- Darwin (macOS + iOS) stat / dirent ABI layout assertions (R13) ---------
//
// The shims write directly into the caller's Darwin `struct stat`/`dirent` via the
// `libc` structs, so the layout is the target's by construction. macOS and iOS share
// the identical 64-bit-inode `stat` (with `st_birthtime` etc.) and BSD `dirent`
// (with `d_seekoff`/`d_namlen`) A?€�t on x86-64 macOS this is the `$INODE64` variant; on
// arm64 macOS and iOS it is simply the native layout (no `$INODE64` suffix exists).
// These pin the *values* synthesized and prove the writes stay in-bounds. They are
// **cross-check-only**: they compile and would run on a real Apple device, but there
// is no Apple host here to execute them.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::undocumented_unsafe_blocks)]
mod layout_tests {
    use super::super::{fill_dirent, fill_stat};
    use crate::vfs::{VfsDirEntry, VfsMetadata};
    use std::path::Path;

    #[test]
    fn fill_stat_reports_a_read_only_regular_file() {
        let meta = VfsMetadata::file(12_345);
        let mut st: libc::stat = unsafe { core::mem::zeroed() };
        unsafe { fill_stat(&raw mut st, Path::new("/__hookfs__/x/clip.braw"), &meta) };
        assert_eq!(st.st_size, 12_345);
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFREG);
        assert_eq!(st.st_mode & 0o777, 0o444);
        assert_eq!(st.st_nlink, 1);
        assert_ne!(st.st_ino, 0, "a stable non-zero inode identity");
        assert_eq!(st.st_blksize, 512);
    }

    #[test]
    fn fill_stat_for_a_directory_sets_ifdir() {
        let meta = VfsMetadata::dir();
        let mut st: libc::stat = unsafe { core::mem::zeroed() };
        unsafe { fill_stat(&raw mut st, Path::new("/d"), &meta) };
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFDIR);
    }

    #[test]
    fn fill_stat_handles_large_sizes() {
        let meta = VfsMetadata::file(1 << 40); // > 4 GiB: exercises the 64-bit fields
        let mut st: libc::stat = unsafe { core::mem::zeroed() };
        unsafe { fill_stat(&raw mut st, Path::new("/big"), &meta) };
        assert_eq!(st.st_size, 1 << 40);
    }

    #[test]
    fn dirent_name_is_nul_terminated_and_truncated() {
        let entry = VfsDirEntry {
            name: std::ffi::OsString::from("sample.braw"),
            metadata: VfsMetadata::file(1),
        };
        let mut d: libc::dirent = unsafe { core::mem::zeroed() };
        let ptr = fill_dirent(&entry, &mut d, 0);
        assert_eq!(ptr, &raw mut d);
        assert_eq!(d.d_type, libc::DT_REG);
        // The macOS `dirent` carries `d_namlen` (glibc does not).
        assert_eq!(usize::from(d.d_namlen), "sample.braw".len());
        let name = unsafe { core::ffi::CStr::from_ptr(d.d_name.as_ptr()) };
        assert_eq!(name.to_str().unwrap(), "sample.braw");
    }
}
