//!
//! The **portable core** (the carrier-fd `open`/`read`/`readv`/`pread`/`lseek`/
//! `close`/write/truncate/mutate family, the stat/dir/path shims, the shared cookie
//! stream logic, and the variadic-`open` trampoline) lives here and compiles on all
//! three targets. The genuinely platform-specific pieces are cfg-split: the cookie
//! API (`fopencookie` vs [`funopen`](macos)), glibc's versioned `__?xstat` + `*64`
//! variants vs Darwin's plain `stat`/`readdir` + `__getdirentries64`, the glibc vs
//! Darwin `struct stat`/`dirent` layouts, and `errno` via `__errno_location` vs
//! `__error`. macOS and iOS share the Darwin path verbatim (identical libc struct
//! layouts and `funopen`); the Darwin-specific entry points live in the [`macos`]
//! submodule.
//!
//! Each shim is an `unsafe extern "C"` function with the *exact* ABI of the libc
//! function it replaces (types cross-checked against `libc`, R13). Every shim:
//!
//! 1. runs its Rust work behind [`guard_abi`] — a panic becomes an `errno` failure,
//!    never an unwind across the FFI boundary (R15);
//! 2. honors the thread-local reentrancy guard — a re-entered shim passes through;
//! 3. routes: the open/path family decides virtual-vs-real by **path**, the fd
//!    family by **registry membership** (fds are generic, so a stray call on a
//!    non-virtual fd passes straight through);
//! 4. for a virtual object, serves it from the VFS/registry and sets `errno` on
//!    every failure path; for anything else, calls the saved **original** pointer so
//!    non-virtual paths and fds keep byte-for-byte / native-error parity.
//!
//! `fopen` on a virtual path returns a **real** `FILE*` built with glibc
//! [`fopencookie`](https://man7.org/linux/man-pages/man3/fopencookie.3.html): the
//! read/seek/close callbacks drive the Rust stream, and `fread`/`fseek`/`ftell`/
//! `fclose`/… deliberately continue into libc, which invokes those callbacks
//! (the **intentional cookie-stream passthroughs** — never hooked). `open` returns
//! a **real carrier fd** (`/dev/null`) reserved through the *original* `open`, so a
//! descriptor that escapes to unhooked code fails predictably.
//!
//! # Variadic `open`
//! `open`/`open64`/`openat` are C-variadic; they are entered through an audited
//! per-arch `global_asm!` tail-call trampoline (see [`trampolines`]) that preserves
//! the optional `mode` argument and lands in a fixed Rust function.

// This module bridges Rust and the libc C ABI on the 64-bit-only supported
// `c_int`/`off_t`/`size_t` and `usize` freely, and those conversions are exact on
// these targets. Allow the scalar-cast lints module-wide rather than sprinkling
// `try_from` through every shim.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

mod trampolines;

// The Darwin-specific entry points (`funopen`, `__getdirentries64`, the
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) mod macos;

use crate::dispatch::{self, HookScope, Sym, current_engine, guard_abi};
use crate::namespace::decode_cstr;
use crate::registry::{OpenFile, OpenStream, VIRTUAL_VOLUME_SERIAL};
use crate::router::{Engine, Route};
use crate::vfs::{OpenOptions, VfsDirEntry, VfsMetadata, VolumeInfo};
use core::ffi::{c_char, c_int, c_void};
use std::io::{self, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::sync::PoisonError;

// Types shared by both libc flavors. `dirent`/`stat` exist on both (with different
// *layouts*, which the fills handle per-platform); `off64_t`/`stat64`/`statx`/
// `dirent64` are glibc-only.
use libc::{DIR, FILE, dirent, mode_t, off_t, size_t, ssize_t, stat, statvfs};
#[cfg(target_os = "linux")]
use libc::{dirent64, off64_t, stat64, statx};

// glibc `fopencookie` and its `cookie_io_functions_t` are not exposed by the
// passed **by value** (four function-pointer fields). macOS uses `funopen` instead
// (see the [`macos`] submodule).
#[cfg(target_os = "linux")]
type CookieRead = unsafe extern "C" fn(*mut c_void, *mut c_char, size_t) -> ssize_t;
#[cfg(target_os = "linux")]
type CookieWrite = unsafe extern "C" fn(*mut c_void, *const c_char, size_t) -> ssize_t;
#[cfg(target_os = "linux")]
type CookieSeek = unsafe extern "C" fn(*mut c_void, *mut off64_t, c_int) -> c_int;
#[cfg(target_os = "linux")]
type CookieClose = unsafe extern "C" fn(*mut c_void) -> c_int;

#[cfg(target_os = "linux")]
#[repr(C)]
struct CookieIoFunctions {
    read: Option<CookieRead>,
    write: Option<CookieWrite>,
    seek: Option<CookieSeek>,
    close: Option<CookieClose>,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    /// glibc `FILE *fopencookie(void *cookie, const char *mode, cookie_io_functions_t io_funcs)`.
    fn fopencookie(
        cookie: *mut c_void,
        mode: *const c_char,
        io_funcs: CookieIoFunctions,
    ) -> *mut FILE;
}

// ---- Original function-pointer types (exact libc ABI) ----------------------

pub(super) type PfnOpen = unsafe extern "C" fn(*const c_char, c_int, ...) -> c_int;
pub(super) type PfnOpenat = unsafe extern "C" fn(c_int, *const c_char, c_int, ...) -> c_int;
type PfnFopen = unsafe extern "C" fn(*const c_char, *const c_char) -> *mut FILE;
type PfnRead = unsafe extern "C" fn(c_int, *mut c_void, size_t) -> ssize_t;
type PfnReadv = unsafe extern "C" fn(c_int, *const libc::iovec, c_int) -> ssize_t;
type PfnPread = unsafe extern "C" fn(c_int, *mut c_void, size_t, off_t) -> ssize_t;
type PfnLseek = unsafe extern "C" fn(c_int, off_t, c_int) -> off_t;
type PfnClose = unsafe extern "C" fn(c_int) -> c_int;
type PfnWrite = unsafe extern "C" fn(c_int, *const c_void, size_t) -> ssize_t;
type PfnWritev = unsafe extern "C" fn(c_int, *const libc::iovec, c_int) -> ssize_t;
type PfnPwrite = unsafe extern "C" fn(c_int, *const c_void, size_t, off_t) -> ssize_t;
type PfnFtruncate = unsafe extern "C" fn(c_int, off_t) -> c_int;
type PfnTruncate = unsafe extern "C" fn(*const c_char, off_t) -> c_int;
type PfnFsync = unsafe extern "C" fn(c_int) -> c_int;
type PfnMkdir = unsafe extern "C" fn(*const c_char, mode_t) -> c_int;
type PfnPathOnly = unsafe extern "C" fn(*const c_char) -> c_int; // remove / unlink / chdir
type PfnRename = unsafe extern "C" fn(*const c_char, *const c_char) -> c_int; // + link / symlink
type PfnFchmod = unsafe extern "C" fn(c_int, mode_t) -> c_int;
type PfnFchmodat = unsafe extern "C" fn(c_int, *const c_char, mode_t, c_int) -> c_int;
type PfnUtimensat =
    unsafe extern "C" fn(c_int, *const c_char, *const libc::timespec, c_int) -> c_int;
type PfnStat = unsafe extern "C" fn(*const c_char, *mut stat) -> c_int;
type PfnFstat = unsafe extern "C" fn(c_int, *mut stat) -> c_int;
type PfnFstatat = unsafe extern "C" fn(c_int, *const c_char, *mut stat, c_int) -> c_int;
type PfnStatvfs = unsafe extern "C" fn(*const c_char, *mut statvfs) -> c_int;
type PfnFstatvfs = unsafe extern "C" fn(c_int, *mut statvfs) -> c_int;
type PfnOpendir = unsafe extern "C" fn(*const c_char) -> *mut DIR;
type PfnFdopendir = unsafe extern "C" fn(c_int) -> *mut DIR;
type PfnReaddir = unsafe extern "C" fn(*mut DIR) -> *mut dirent;
type PfnClosedir = unsafe extern "C" fn(*mut DIR) -> c_int;
type PfnRealpath = unsafe extern "C" fn(*const c_char, *mut c_char) -> *mut c_char;
type PfnAccess = unsafe extern "C" fn(*const c_char, c_int) -> c_int;
type PfnFaccessat = unsafe extern "C" fn(c_int, *const c_char, c_int, c_int) -> c_int;
type PfnGetcwd = unsafe extern "C" fn(*mut c_char, size_t) -> *mut c_char;
type PfnReadlink = unsafe extern "C" fn(*const c_char, *mut c_char, size_t) -> ssize_t;
type PfnDlopen = unsafe extern "C" fn(*const c_char, c_int) -> *mut c_void;
type PfnDlclose = unsafe extern "C" fn(*mut c_void) -> c_int;

// glibc-only entry points (macOS has no `*64` variants, no versioned `__?xstat`, no
// `statx`, and a differently-typed `sendfile`).
#[cfg(target_os = "linux")]
type PfnLseek64 = unsafe extern "C" fn(c_int, off64_t, c_int) -> off64_t;
#[cfg(target_os = "linux")]
type PfnFtruncate64 = unsafe extern "C" fn(c_int, off64_t) -> c_int;
#[cfg(target_os = "linux")]
type PfnSendfile = unsafe extern "C" fn(c_int, c_int, *mut off_t, size_t) -> ssize_t;
#[cfg(target_os = "linux")]
type PfnXstat = unsafe extern "C" fn(c_int, *const c_char, *mut stat) -> c_int;
#[cfg(target_os = "linux")]
type PfnFxstat = unsafe extern "C" fn(c_int, c_int, *mut stat) -> c_int;
#[cfg(target_os = "linux")]
type PfnXstat64 = unsafe extern "C" fn(c_int, *const c_char, *mut stat64) -> c_int;
#[cfg(target_os = "linux")]
type PfnFxstat64 = unsafe extern "C" fn(c_int, c_int, *mut stat64) -> c_int;
#[cfg(target_os = "linux")]
type PfnStatx =
    unsafe extern "C" fn(c_int, *const c_char, c_int, libc::c_uint, *mut statx) -> c_int;
#[cfg(target_os = "linux")]
type PfnReaddir64 = unsafe extern "C" fn(*mut DIR) -> *mut dirent64;

/// The `errno` used when a shim's Rust work panics (R15).
const PANIC_ERRNO: u32 = libc::EIO as u32;

// ---- Small helpers ---------------------------------------------------------

/// Transmute the saved original of `sym` into its function-pointer type.
#[inline]
pub(super) fn orig_fn<T: Copy>(sym: Sym) -> T {
    let ptr = dispatch::original(sym);
    debug_assert_ne!(
        ptr, 0,
        "original for {sym:?} was not resolved before the hook ran"
    );
    // SAFETY: `T` is an `extern "C"` fn-pointer type (pointer-sized) and `ptr` is
    // the canonical libc export resolved at install time.
    unsafe { core::mem::transmute_copy::<usize, T>(&ptr) }
}

/// Set the thread-local `errno` via the platform accessor: glibc's
#[inline]
pub(super) fn set_errno(code: c_int) {
    #[cfg(target_os = "linux")]
    // SAFETY: `__errno_location` returns a valid, thread-local `int*`.
    unsafe {
        *libc::__errno_location() = code;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    // SAFETY: `__error` returns a valid, thread-local `int*`.
    unsafe {
        *libc::__error() = code;
    }
}

/// Map an `io::Error` to the closest `errno`.
pub(super) fn io_to_errno(err: &io::Error) -> c_int {
    err.raw_os_error().unwrap_or(match err.kind() {
        io::ErrorKind::NotFound => libc::ENOENT,
        io::ErrorKind::PermissionDenied => libc::EACCES,
        io::ErrorKind::AlreadyExists => libc::EEXIST,
        io::ErrorKind::InvalidInput => libc::EINVAL,
        io::ErrorKind::DirectoryNotEmpty => libc::ENOTEMPTY,
        io::ErrorKind::NotADirectory => libc::ENOTDIR,
        io::ErrorKind::IsADirectory => libc::EISDIR,
        _ => libc::EIO,
    })
}

/// Lock an open-file entry, recovering a poisoned lock (a panicked prior holder
/// left no torn state — the shim just serves the next call).
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Route a decoded path via the active engine. `None` = no engine / re-entered.
pub(super) fn route(decoded: &str) -> Option<(Arc<Engine>, Route)> {
    let engine = current_engine()?;
    let route = engine.classify(decoded);
    Some((engine, route))
}

/// A stable 64-bit inode identity for a virtual path (so repeated opens report the
/// same `st_ino`). Deterministic, derived from the normalized comparison key.
fn stable_ino(path: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    crate::namespace::normalize_key(path).hash(&mut hasher);
    match hasher.finish() {
        0 => 1,
        id => id,
    }
}

/// Open a real, harmless carrier fd (`/dev/null`) through the *original* `open`,
/// reserving a collision-free positive descriptor. Returns `None` on failure.
fn carrier_fd() -> Option<c_int> {
    let open: PfnOpen = orig_fn(Sym::Open);
    let path = c"/dev/null";
    // SAFETY: valid NUL-terminated path; `O_RDONLY`; the variadic call passes no
    // `mode` (correct, since `O_CREAT` is absent).
    let fd = unsafe { open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    (fd >= 0).then_some(fd)
}

/// The `(mode, size, mtime_secs)` a virtual `stat` should report.
struct StatValues {
    mode: mode_t,
    size: i64,
    mtime: i64,
}

fn stat_values(meta: &VfsMetadata) -> StatValues {
    let mode: mode_t = if meta.is_dir {
        libc::S_IFDIR | 0o555
    } else {
        libc::S_IFREG | 0o444
    };
    let size = i64::try_from(meta.len).unwrap_or(i64::MAX);
    let mtime = meta
        .mtime
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0));
    StatValues { mode, size, mtime }
}

/// The 512-byte block count (`st_blocks`/`stx_blocks`) a reported byte length
/// should carry. Computed in `u64` with a saturating add so it can never overflow
/// — `size` is a non-negative length already clamped to `i64::MAX`, and the naive
/// `i64` form `(size + 511) / 512` would overflow at that clamp.
fn blocks_512(size: i64) -> u64 {
    (size as u64).saturating_add(511) / 512
}

/// Fill a `struct stat` from VFS metadata.
///
/// # Safety
/// `buf` must be a valid, writable `*mut stat`.
unsafe fn fill_stat(buf: *mut stat, path: &Path, meta: &VfsMetadata) {
    if buf.is_null() {
        return;
    }
    let v = stat_values(meta);
    // SAFETY: `stat` is plain-old-data; an all-zero bit pattern is valid.
    let mut st: stat = unsafe { core::mem::zeroed() };
    // `dev_t` differs (`u64` on glibc, `i32` on macOS); `try_from` fits both (the
    // conversion is infallible on glibc, hence the Linux-only allow). The remaining
    // fields exist with the same names on both layouts (macOS adds `st_birthtime`
    // etc., left zero); the macOS `stat` is the 64-bit-inode layout (`$INODE64`),
    #[cfg_attr(target_os = "linux", allow(clippy::unnecessary_fallible_conversions))]
    {
        st.st_dev = libc::dev_t::try_from(VIRTUAL_VOLUME_SERIAL).unwrap_or(0);
    }
    st.st_ino = stable_ino(path);
    st.st_mode = v.mode;
    st.st_nlink = 1;
    st.st_size = v.size as off_t;
    st.st_blksize = 512;
    st.st_blocks = blocks_512(v.size) as libc::blkcnt_t;
    st.st_mtime = v.mtime as libc::time_t;
    st.st_atime = v.mtime as libc::time_t;
    st.st_ctime = v.mtime as libc::time_t;
    // SAFETY: caller guarantees `buf` is writable.
    unsafe { *buf = st };
}

/// Fill a `struct stat64` from VFS metadata (glibc's `*64` stat family).
///
/// # Safety
/// `buf` must be a valid, writable `*mut stat64`.
#[cfg(target_os = "linux")]
unsafe fn fill_stat64(buf: *mut stat64, path: &Path, meta: &VfsMetadata) {
    if buf.is_null() {
        return;
    }
    let v = stat_values(meta);
    // SAFETY: `stat64` is plain-old-data; all-zero is a valid bit pattern.
    let mut st: stat64 = unsafe { core::mem::zeroed() };
    st.st_dev = u64::from(VIRTUAL_VOLUME_SERIAL);
    st.st_ino = stable_ino(path);
    st.st_mode = v.mode;
    st.st_nlink = 1;
    st.st_size = v.size;
    st.st_blksize = 512;
    st.st_blocks = blocks_512(v.size) as libc::blkcnt64_t;
    st.st_mtime = v.mtime as libc::time_t;
    st.st_atime = v.mtime as libc::time_t;
    st.st_ctime = v.mtime as libc::time_t;
    // SAFETY: caller guarantees `buf` is writable.
    unsafe { *buf = st };
}

/// Fill a `struct statx` from VFS metadata (Linux `statx`).
///
/// # Safety
/// `buf` must be a valid, writable `*mut statx`.
#[cfg(target_os = "linux")]
unsafe fn fill_statx(buf: *mut statx, path: &Path, meta: &VfsMetadata) {
    if buf.is_null() {
        return;
    }
    let v = stat_values(meta);
    // SAFETY: `statx` is plain-old-data; all-zero is a valid bit pattern.
    let mut st: statx = unsafe { core::mem::zeroed() };
    st.stx_mask = libc::STATX_BASIC_STATS;
    st.stx_blksize = 512;
    st.stx_nlink = 1;
    st.stx_mode = v.mode as u16;
    st.stx_ino = stable_ino(path);
    st.stx_size = u64::try_from(v.size).unwrap_or(0);
    st.stx_blocks = blocks_512(v.size);
    st.stx_mtime.tv_sec = v.mtime;
    st.stx_atime.tv_sec = v.mtime;
    st.stx_ctime.tv_sec = v.mtime;
    // SAFETY: caller guarantees `buf` is writable.
    unsafe { *buf = st };
}

// ---- Cookie streams (shared: fopencookie / funopen) ------------------------

/// The state behind a virtual `FILE*`: the read/write stream driving the cookie
/// callbacks. Boxed and passed to `fopencookie` (Linux) / `funopen` (macOS) as the
/// opaque cookie; reclaimed by the close callback. `writable` decides whether the
/// platform backend installs a write callback (a read-only cookie has none, so a
pub(crate) struct CookieState {
    stream: OpenStream,
    writable: bool,
}

/// Force the `IN_HOOK` reentrancy guard set for the extent of the returned scope,
///
/// A cookie callback drives the provider stream and — unlike an fd-path shim — has
/// no original to defer to, so it must always run its body. Rather than the shims'
/// *enter-or-passthrough* guard (which returns to the original when already inside a
/// hook), a cookie callback **force-sets** `IN_HOOK` around the provider call so a
/// backing `FileStream` that performs real I/O through a patched module passes
/// through instead of re-dispatching. [`HookScope::enter`] returns `Some` — whose
/// `Drop` clears the flag — only when *this* call set it (the thread was not already
/// in a hook); when the thread was already inside a hook it returns `None` and the
/// flag stays set. Either way the scope runs with `IN_HOOK` set and the prior value
/// is restored on drop, including when the provider panics (the enclosing
/// [`guard_abi`] catches the unwind after the guard has run).
#[must_use]
pub(crate) fn force_in_hook() -> Option<HookScope> {
    HookScope::enter()
}

// The three cookie operations are shared by both platforms' callbacks (which differ
// only in their C ABI: `fopencookie`'s `size_t`/`off64_t*` vs `funopen`'s
// `int`/`fpos_t`). They run inside the callback's `guard_abi`, so a provider panic
// is already contained. `force_in_hook` passes a backing stream's real I/O through.

/// Copy up to `len` bytes from the cookie's stream into `buf`, returning the byte
/// count or `-1` with `errno` set. `buf` must address `len` writable bytes.
pub(crate) fn cookie_read_into(cookie: *mut c_void, buf: *mut u8, len: usize) -> isize {
    if cookie.is_null() || (buf.is_null() && len != 0) {
        set_errno(libc::EINVAL);
        return -1;
    }
    // SAFETY: `cookie` is the `CookieState` handed to `fopencookie`/`funopen`.
    let state = unsafe { &mut *cookie.cast::<CookieState>() };
    if len == 0 {
        return 0;
    }
    // SAFETY: the cookie API guarantees `buf` addresses `len` writable bytes.
    let dst = unsafe { core::slice::from_raw_parts_mut(buf, len) };
    let read = {
        let _in_hook = force_in_hook();
        state.stream.read(dst)
    };
    match read {
        Ok(n) => isize::try_from(n).unwrap_or(isize::MAX),
        Err(err) => {
            set_errno(io_to_errno(&err));
            -1
        }
    }
}

/// Copy `len` bytes from `buf` into the cookie's stream at its cursor (growing the
/// file), returning the byte count or `-1` with `errno` set. `buf` must address
/// `len` readable bytes.
pub(crate) fn cookie_write_from(cookie: *mut c_void, buf: *const u8, len: usize) -> isize {
    if cookie.is_null() || (buf.is_null() && len != 0) {
        set_errno(libc::EINVAL);
        return -1;
    }
    // SAFETY: `cookie` is the `CookieState` handed to `fopencookie`/`funopen`.
    let state = unsafe { &mut *cookie.cast::<CookieState>() };
    if len == 0 {
        return 0;
    }
    // SAFETY: the cookie API guarantees `buf` addresses `len` readable bytes.
    let src = unsafe { core::slice::from_raw_parts(buf, len) };
    let written = {
        let _in_hook = force_in_hook();
        state.stream.write(src)
    };
    match written {
        Ok(n) => isize::try_from(n).unwrap_or(isize::MAX),
        Err(err) => {
            set_errno(io_to_errno(&err));
            -1
        }
    }
}

/// Seek the cookie's stream, returning the new absolute offset, or `Err(())` with
/// `errno` set.
pub(crate) fn cookie_seek_to(cookie: *mut c_void, offset: i64, whence: c_int) -> Result<i64, ()> {
    if cookie.is_null() {
        set_errno(libc::EINVAL);
        return Err(());
    }
    // SAFETY: `cookie` is our `CookieState`.
    let state = unsafe { &mut *cookie.cast::<CookieState>() };
    let Some(from) = seek_from(whence, offset) else {
        set_errno(libc::EINVAL);
        return Err(());
    };
    let seeked = {
        let _in_hook = force_in_hook();
        state.stream.seek(from)
    };
    match seeked {
        Ok(pos) => Ok(i64::try_from(pos).unwrap_or(i64::MAX)),
        Err(err) => {
            set_errno(io_to_errno(&err));
            Err(())
        }
    }
}

/// Reclaim the boxed cookie state exactly once (the close callback).
pub(crate) fn cookie_close_state(cookie: *mut c_void) {
    if cookie.is_null() {
        return;
    }
    // SAFETY: `cookie` is the `Box<CookieState>` handed to the cookie API, closed
    // exactly once by libc.
    let state = unsafe { Box::from_raw(cookie.cast::<CookieState>()) };
    // Dropping the provider stream may perform real I/O (e.g. closing a backing fd)
    let _in_hook = force_in_hook();
    drop(state);
}

/// `fopencookie` read callback (Linux) — adapts the shared logic to the glibc ABI.
#[cfg(target_os = "linux")]
extern "C" fn cookie_read(cookie: *mut c_void, buf: *mut c_char, size: size_t) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        cookie_read_into(cookie, buf.cast::<u8>(), size) as ssize_t
    })
}

#[cfg(target_os = "linux")]
extern "C" fn cookie_write(cookie: *mut c_void, buf: *const c_char, size: size_t) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        cookie_write_from(cookie, buf.cast::<u8>(), size) as ssize_t
    })
}

/// `fopencookie` seek callback (Linux) — new offset returned via the in/out pointer.
#[cfg(target_os = "linux")]
extern "C" fn cookie_seek(cookie: *mut c_void, offset: *mut off64_t, whence: c_int) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        if offset.is_null() {
            set_errno(libc::EINVAL);
            return -1;
        }
        // SAFETY: `offset` is a valid, readable/writable `*mut off64_t`.
        let requested = unsafe { *offset };
        match cookie_seek_to(cookie, requested, whence) {
            Ok(pos) => {
                // SAFETY: `offset` is writable.
                unsafe { *offset = pos };
                0
            }
            Err(()) => -1,
        }
    })
}

/// `fopencookie` close callback (Linux).
#[cfg(target_os = "linux")]
extern "C" fn cookie_close(cookie: *mut c_void) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        cookie_close_state(cookie);
        0
    })
}

/// Translate a `fopen` mode string into [`OpenOptions`] per the glibc grammar
/// (`libio/fileops.c`). The first character selects the base access — `r` opens
/// read-only, `w` creates+truncates for writing, `a` creates+appends for writing —
/// and any other first character (or an empty string) is invalid (`None`; glibc
/// returns `EINVAL`). A `+` among the flags upgrades to read-write. The modifier
/// letters (`b` binary, `t` text, `x` `O_EXCL`, `e` `O_CLOEXEC`, `m` mmap, `c`
/// no-cancel) do not by themselves change read-vs-write intent; glibc's trailing
/// `,ccs=CHARSET` selector is excluded before the scan.
fn fopen_options(mode: &str) -> Option<OpenOptions> {
    // The access flags precede an optional `,ccs=CHARSET` charset selector.
    let flags = mode.split_once(',').map_or(mode, |(head, _)| head);
    let plus = flags.contains('+');
    let create_new = flags.contains('x'); // glibc `x` == O_EXCL (fail if it exists)
    match flags.chars().next()? {
        'r' => Some(OpenOptions {
            read: true,
            write: plus,
            ..OpenOptions::default()
        }),
        'w' => Some(OpenOptions {
            read: plus,
            write: true,
            create: true,
            create_new,
            truncate: true,
            ..OpenOptions::default()
        }),
        'a' => Some(OpenOptions {
            read: plus,
            write: true,
            create: true,
            create_new,
            append: true,
            ..OpenOptions::default()
        }),
        _ => None,
    }
}

/// Parse + validate the `fopen` mode, gate writes, open the VFS stream (read or
/// write per the mode), and box it as a [`CookieState`] ready for `fopencookie`/
/// `funopen`. Returns the raw cookie pointer on success (the caller owns it and must
/// hand it to the cookie API or free it), or `None` with `errno` set — invalid mode
/// (`EINVAL`), a write mode when writes are disallowed (`EROFS`, the same gate as
/// [`open_virtual_fd`]), a missing virtual path (`ENOENT`), or a provider error.
/// Shared by both platforms' `fopen`.
pub(crate) fn prepare_cookie(
    engine: &Engine,
    path: &Path,
    mode: *const c_char,
) -> Option<*mut CookieState> {
    // SAFETY: `mode` is the caller's NUL-terminated mode string (or null; `fopen`
    // with a null mode is invalid).
    let Some(mode_str) = (unsafe { decode_cstr(mode) }) else {
        set_errno(libc::EINVAL);
        return None;
    };
    let Some(opts) = fopen_options(&mode_str) else {
        set_errno(libc::EINVAL);
        return None;
    };
    if opts.write && !engine.allow_writes() {
        set_errno(libc::EROFS); // writes disabled: fail at open, not at write
        return None;
    }
    let opened = if opts.write {
        engine
            .vfs()
            .open_write(path, &opts)
            .map(|r| r.map(OpenStream::Write))
    } else {
        engine
            .vfs()
            .open(path, &opts)
            .map(|r| r.map(OpenStream::Read))
    };
    match opened {
        None => {
            set_errno(libc::ENOENT);
            None
        }
        Some(Err(err)) => {
            set_errno(io_to_errno(&err));
            None
        }
        Some(Ok(stream)) => Some(Box::into_raw(Box::new(CookieState {
            stream,
            writable: opts.write,
        }))),
    }
}

/// Open a virtual path as a real `fopencookie`-backed `FILE*` (Linux). A read-only
/// stream has a **null write callback**, so any `fwrite`/`fprintf` fails predictably
/// [`cookie_write`], so `fwrite`/`fprintf` reach the VFS. Returns null with `errno`
/// on failure.
#[cfg(target_os = "linux")]
fn fopen_virtual(engine: &Engine, path: &Path, mode: *const c_char) -> *mut FILE {
    let Some(cookie) = prepare_cookie(engine, path, mode) else {
        return core::ptr::null_mut();
    };
    // SAFETY: `cookie` is the live `CookieState` just boxed by `prepare_cookie`.
    let writable = unsafe { (*cookie).writable };
    let io_funcs = CookieIoFunctions {
        read: Some(cookie_read),
        write: writable.then_some(cookie_write as CookieWrite),
        seek: Some(cookie_seek),
        close: Some(cookie_close),
    };
    // SAFETY: `cookie` is a live `CookieState`; `mode` is the caller's mode string;
    // `io_funcs` holds the callbacks above.
    let file = unsafe { fopencookie(cookie.cast::<c_void>(), mode, io_funcs) };
    if file.is_null() {
        // fopencookie failed: reclaim the cookie it never took ownership of.
        // SAFETY: `cookie` was not consumed by the failed `fopencookie`.
        drop(unsafe { Box::from_raw(cookie) });
        set_errno(libc::EIO);
    }
    file
}

// ---- Open family (path-routed) ---------------------------------------------

/// Shared virtual `open` logic: allocate a carrier fd bound to a virtual open,
/// honoring `O_WRONLY`/`O_RDWR`/`O_CREAT`/`O_EXCL`/`O_TRUNC`/`O_APPEND`. Returns the
/// carrier fd, or `-1` with `errno` set (fail-closed). A write open with writes
/// disabled fails `EROFS`.
pub(super) fn open_virtual_fd(engine: &Engine, path: &Path, flags: c_int) -> c_int {
    let write = flags & (libc::O_WRONLY | libc::O_RDWR) != 0;
    if write && !engine.allow_writes() {
        set_errno(libc::EROFS);
        return -1;
    }
    let opts = OpenOptions {
        // `O_RDONLY` (0) and `O_RDWR` grant read; `O_WRONLY` does not.
        read: flags & libc::O_WRONLY == 0,
        write,
        create: flags & libc::O_CREAT != 0,
        create_new: flags & (libc::O_CREAT | libc::O_EXCL) == (libc::O_CREAT | libc::O_EXCL),
        truncate: flags & libc::O_TRUNC != 0,
        append: flags & libc::O_APPEND != 0,
    };
    let opened = if write {
        engine
            .vfs()
            .open_write(path, &opts)
            .map(|r| r.map(OpenStream::Write))
    } else {
        engine
            .vfs()
            .open(path, &opts)
            .map(|r| r.map(OpenStream::Read))
    };
    match opened {
        None => {
            set_errno(libc::ENOENT); // virtual path the VFS does not have: fail closed
            -1
        }
        Some(Err(err)) => {
            set_errno(io_to_errno(&err));
            -1
        }
        Some(Ok(stream)) => {
            let Some(fd) = carrier_fd() else {
                set_errno(libc::EMFILE);
                return -1;
            };
            engine.registry().insert_file(
                fd as usize,
                OpenFile {
                    stream,
                    path: path.to_owned(),
                    writable: write,
                },
            );
            fd
        }
    }
}

/// Dispatch a virtual `fopen` to the platform's cookie backend (`fopencookie` on
/// Linux, `funopen` on macOS) — the only platform-specific part of the shared
/// [`shim_fopen`] routing.
#[cfg(target_os = "linux")]
fn fopen_virtual_dispatch(engine: &Engine, path: &Path, mode: *const c_char) -> *mut FILE {
    fopen_virtual(engine, path, mode)
}
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn fopen_virtual_dispatch(engine: &Engine, path: &Path, mode: *const c_char) -> *mut FILE {
    macos::fopen_virtual(engine, path, mode)
}

/// `fopen` shim — returns a real cookie-backed `FILE*` for a virtual path. Routing is
/// shared; the cookie backend is platform-specific ([`fopen_virtual_dispatch`]).
pub(crate) extern "C" fn shim_fopen(path: *const c_char, mode: *const c_char) -> *mut FILE {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFopen = orig_fn(Sym::Fopen);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, mode) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: `path` is the caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                core::ptr::null_mut()
            }
            Some((engine, Route::Virtual(p))) => fopen_virtual_dispatch(&engine, &p, mode),
        }
    })
}

/// `fopen64` shim (Linux) — identical routing; 64-bit `off_t` is the default on
/// these targets. macOS has no `fopen64`.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_fopen64(path: *const c_char, mode: *const c_char) -> *mut FILE {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFopen = orig_fn(Sym::Fopen64);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, mode) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                core::ptr::null_mut()
            }
            Some((engine, Route::Virtual(p))) => fopen_virtual(&engine, &p, mode),
        }
    })
}

// ---- Descriptor family (registry-routed) -----------------------------------

/// `read` shim.
pub(crate) extern "C" fn shim_read(fd: c_int, buf: *mut c_void, count: size_t) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnRead = orig_fn(Sym::Read);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, buf, count) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(fd as usize) else {
            return pass();
        };
        if count == 0 {
            return 0;
        }
        if buf.is_null() {
            set_errno(libc::EFAULT);
            return -1;
        }
        let mut open = lock(&file);
        // SAFETY: `buf` is non-null (checked) and the caller guarantees it
        // addresses `count` writable bytes.
        let dst = unsafe { core::slice::from_raw_parts_mut(buf.cast::<u8>(), count) };
        match open.stream.read(dst) {
            Ok(n) => n as ssize_t,
            Err(err) => {
                set_errno(io_to_errno(&err));
                -1
            }
        }
    })
}

/// `readv` shim — fill the iovecs in order from the virtual stream.
pub(crate) extern "C" fn shim_readv(fd: c_int, iov: *const libc::iovec, iovcnt: c_int) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnReadv = orig_fn(Sym::Readv);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, iov, iovcnt) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(fd as usize) else {
            return pass();
        };
        if iov.is_null() || iovcnt < 0 {
            set_errno(libc::EINVAL);
            return -1;
        }
        let mut open = lock(&file);
        let mut total: ssize_t = 0;
        for i in 0..iovcnt as usize {
            // SAFETY: the caller guarantees `iov` addresses `iovcnt` iovecs.
            let v = unsafe { &*iov.add(i) };
            if v.iov_len == 0 || v.iov_base.is_null() {
                continue;
            }
            // SAFETY: each iovec describes `iov_len` writable bytes.
            let dst =
                unsafe { core::slice::from_raw_parts_mut(v.iov_base.cast::<u8>(), v.iov_len) };
            match open.stream.read(dst) {
                Ok(0) => break,
                Ok(n) => {
                    total += n as ssize_t;
                    if n < v.iov_len {
                        break; // short read: stop, as the kernel would
                    }
                }
                Err(err) => {
                    if total == 0 {
                        set_errno(io_to_errno(&err));
                        return -1;
                    }
                    break;
                }
            }
        }
        total
    })
}

/// `pread` shim — positioned read that does not disturb the stream cursor.
pub(crate) extern "C" fn shim_pread(
    fd: c_int,
    buf: *mut c_void,
    count: size_t,
    offset: off_t,
) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnPread = orig_fn(Sym::Pread);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, buf, count, offset) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(fd as usize) else {
            return pass();
        };
        if count == 0 {
            return 0;
        }
        if buf.is_null() || offset < 0 {
            set_errno(if buf.is_null() {
                libc::EFAULT
            } else {
                libc::EINVAL
            });
            return -1;
        }
        let mut open = lock(&file);
        let saved = match open.stream.stream_position() {
            Ok(p) => p,
            Err(err) => {
                set_errno(io_to_errno(&err));
                return -1;
            }
        };
        // SAFETY: `buf` is non-null and addresses `count` writable bytes.
        let dst = unsafe { core::slice::from_raw_parts_mut(buf.cast::<u8>(), count) };
        let result = open
            .stream
            .seek(SeekFrom::Start(offset as u64))
            .and_then(|_| open.stream.read(dst));
        // pread must not change the file position.
        let _ = open.stream.seek(SeekFrom::Start(saved));
        match result {
            Ok(n) => n as ssize_t,
            Err(err) => {
                set_errno(io_to_errno(&err));
                -1
            }
        }
    })
}

/// Translate a POSIX `whence` + offset into a [`SeekFrom`].
fn seek_from(whence: c_int, offset: i64) -> Option<SeekFrom> {
    match whence {
        libc::SEEK_SET => u64::try_from(offset).ok().map(SeekFrom::Start),
        libc::SEEK_CUR => Some(SeekFrom::Current(offset)),
        libc::SEEK_END => Some(SeekFrom::End(offset)),
        _ => None,
    }
}

/// Shared `lseek`/`lseek64` logic.
fn lseek_common(sym: Sym, fd: c_int, offset: i64, whence: c_int) -> i64 {
    let Some(engine) = current_engine() else {
        return lseek_pass(sym, fd, offset, whence);
    };
    let Some(file) = engine.registry().get_file(fd as usize) else {
        return lseek_pass(sym, fd, offset, whence);
    };
    let Some(from) = seek_from(whence, offset) else {
        set_errno(libc::EINVAL);
        return -1;
    };
    let mut open = lock(&file);
    match open.stream.seek(from) {
        Ok(pos) => i64::try_from(pos).unwrap_or(i64::MAX),
        Err(err) => {
            set_errno(io_to_errno(&err));
            -1
        }
    }
}

fn lseek_pass(sym: Sym, fd: c_int, offset: i64, whence: c_int) -> i64 {
    match sym {
        // glibc's `lseek64` takes/returns `off64_t` (also `i64`); macOS has no such
        // variant, so this arm compiles only on Linux.
        #[cfg(target_os = "linux")]
        Sym::Lseek64 => {
            let orig: PfnLseek64 = orig_fn(Sym::Lseek64);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, offset, whence) }
        }
        _ => {
            let orig: PfnLseek = orig_fn(Sym::Lseek);
            // SAFETY: forwarding the caller's exact arguments (`off_t` is `i64`).
            unsafe { orig(fd, offset as off_t, whence) }
        }
    }
}

/// `lseek` shim.
pub(crate) extern "C" fn shim_lseek(fd: c_int, offset: off_t, whence: c_int) -> off_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let Some(_scope) = HookScope::enter() else {
            return lseek_pass(Sym::Lseek, fd, offset, whence) as off_t;
        };
        lseek_common(Sym::Lseek, fd, offset, whence) as off_t
    })
}

/// `lseek64` shim (Linux; macOS `lseek` is already 64-bit).
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_lseek64(fd: c_int, offset: off64_t, whence: c_int) -> off64_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let Some(_scope) = HookScope::enter() else {
            return lseek_pass(Sym::Lseek64, fd, offset, whence);
        };
        lseek_common(Sym::Lseek64, fd, offset, whence)
    })
}

/// `close` shim — drop the registry entry once, then close the carrier fd.
pub(crate) extern "C" fn shim_close(fd: c_int) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnClose = orig_fn(Sym::Close);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(fd) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if let Some(file) = engine.registry().remove_file(fd as usize) {
            drop(file); // release the stream/source Arc
            // Close the real carrier fd (its value == fd) via the original.
            let orig: PfnClose = orig_fn(Sym::Close);
            // SAFETY: `fd` is our carrier fd; closed exactly once.
            unsafe { orig(fd) }
        } else {
            pass()
        }
    })
}

// ---- Write / truncate / mutate — fail closed (EROFS) -----------------------

/// Whether a write to `fd` must be denied because it targets a **non-writable**
/// virtual open. A registry hit is a virtual carrier fd; the write is denied unless
/// the open was granted write access — always the case in the read-only milestone,
/// where every virtual open is non-writable, so every virtual write fails closed.
/// The per-file lock read is purposeful (it consults the open's `writable` flag, the
/// same signal Windows reports as `FILE_ATTRIBUTE_READONLY`), and it is the seam the
fn deny_write_fd(engine: &Engine, fd: c_int) -> bool {
    engine.registry().get_file(fd as usize).is_some_and(|file| {
        !file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .writable
    })
}

/// `write` shim — writes to a writable virtual fd; `EROFS` for a read-only virtual
/// fd; passthrough otherwise.
pub(crate) extern "C" fn shim_write(fd: c_int, buf: *const c_void, count: size_t) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnWrite = orig_fn(Sym::Write);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, buf, count) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(fd as usize) else {
            return pass();
        };
        let mut open = lock(&file);
        if !open.writable {
            set_errno(libc::EROFS);
            return -1;
        }
        if count == 0 {
            return 0;
        }
        if buf.is_null() {
            set_errno(libc::EFAULT);
            return -1;
        }
        // SAFETY: `buf` is non-null (checked) and addresses `count` readable bytes.
        let src = unsafe { core::slice::from_raw_parts(buf.cast::<u8>(), count) };
        match open.stream.write(src) {
            Ok(n) => n as ssize_t,
            Err(err) => {
                set_errno(io_to_errno(&err));
                -1
            }
        }
    })
}

/// `writev` shim — gather-write each iovec to a writable virtual fd; `EROFS` for a
/// read-only virtual fd; passthrough otherwise.
pub(crate) extern "C" fn shim_writev(fd: c_int, iov: *const libc::iovec, iovcnt: c_int) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnWritev = orig_fn(Sym::Writev);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, iov, iovcnt) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(fd as usize) else {
            return pass();
        };
        let mut open = lock(&file);
        if !open.writable {
            set_errno(libc::EROFS);
            return -1;
        }
        if iov.is_null() || iovcnt < 0 {
            set_errno(libc::EINVAL);
            return -1;
        }
        let mut total: ssize_t = 0;
        for i in 0..iovcnt as usize {
            // SAFETY: the caller guarantees `iov` addresses `iovcnt` iovecs.
            let v = unsafe { &*iov.add(i) };
            if v.iov_len == 0 || v.iov_base.is_null() {
                continue;
            }
            // SAFETY: each iovec describes `iov_len` readable bytes.
            let src = unsafe { core::slice::from_raw_parts(v.iov_base.cast::<u8>(), v.iov_len) };
            match open.stream.write(src) {
                Ok(n) => {
                    total += n as ssize_t;
                    if n < v.iov_len {
                        break; // short write: stop, as the kernel would
                    }
                }
                Err(err) => {
                    if total == 0 {
                        set_errno(io_to_errno(&err));
                        return -1;
                    }
                    break;
                }
            }
        }
        total
    })
}

/// `pwrite` shim — positioned write to a writable virtual fd that does not disturb
/// the stream cursor; `EROFS` for a read-only virtual fd; passthrough otherwise.
pub(crate) extern "C" fn shim_pwrite(
    fd: c_int,
    buf: *const c_void,
    count: size_t,
    offset: off_t,
) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnPwrite = orig_fn(Sym::Pwrite);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, buf, count, offset) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(fd as usize) else {
            return pass();
        };
        let mut open = lock(&file);
        if !open.writable {
            set_errno(libc::EROFS);
            return -1;
        }
        if count == 0 {
            return 0;
        }
        if buf.is_null() || offset < 0 {
            set_errno(if buf.is_null() {
                libc::EFAULT
            } else {
                libc::EINVAL
            });
            return -1;
        }
        let saved = match open.stream.stream_position() {
            Ok(p) => p,
            Err(err) => {
                set_errno(io_to_errno(&err));
                return -1;
            }
        };
        // SAFETY: `buf` is non-null and addresses `count` readable bytes.
        let src = unsafe { core::slice::from_raw_parts(buf.cast::<u8>(), count) };
        let result = open
            .stream
            .seek(SeekFrom::Start(offset as u64))
            .and_then(|_| open.stream.write(src));
        // pwrite must not change the file position.
        let _ = open.stream.seek(SeekFrom::Start(saved));
        match result {
            Ok(n) => n as ssize_t,
            Err(err) => {
                set_errno(io_to_errno(&err));
                -1
            }
        }
    })
}

/// `ftruncate` shim — truncate/extend a writable virtual fd; `EROFS` for a read-only
/// virtual fd; passthrough otherwise.
pub(crate) extern "C" fn shim_ftruncate(fd: c_int, length: off_t) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFtruncate = orig_fn(Sym::Ftruncate);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, length) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        ftruncate_fd(&engine, fd, length, pass)
    })
}

/// `ftruncate64` shim (Linux) — truncate/extend a writable virtual fd; `EROFS` for a
/// read-only virtual fd; passthrough otherwise.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_ftruncate64(fd: c_int, length: off64_t) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFtruncate64 = orig_fn(Sym::Ftruncate64);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, length) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        ftruncate_fd(&engine, fd, length, pass)
    })
}

/// Shared `ftruncate`/`ftruncate64` logic: set the length of a writable virtual fd,
/// deny a read-only one (`EROFS`), or passthrough a non-virtual fd.
fn ftruncate_fd(engine: &Engine, fd: c_int, length: i64, pass: impl FnOnce() -> c_int) -> c_int {
    let Some(file) = engine.registry().get_file(fd as usize) else {
        return pass();
    };
    let mut open = lock(&file);
    if !open.writable {
        set_errno(libc::EROFS);
        return -1;
    }
    if length < 0 {
        set_errno(libc::EINVAL);
        return -1;
    }
    match open.stream.set_len(length as u64) {
        Ok(()) => 0,
        Err(err) => {
            set_errno(io_to_errno(&err));
            -1
        }
    }
}

/// `fsync`/`fdatasync` shim — flush a virtual fd (a no-op success for a read-only
/// one); passthrough for a non-virtual fd.
pub(crate) extern "C" fn shim_fsync(fd: c_int) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFsync = orig_fn(Sym::Fsync);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(fd) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(fd as usize) else {
            return pass();
        };
        let mut open = lock(&file);
        match open.stream.flush() {
            Ok(()) => 0,
            Err(err) => {
                set_errno(io_to_errno(&err));
                -1
            }
        }
    })
}

/// A path-routed **attribute** mutation that stays denied (`EROFS`) for a virtual
/// path even when writes are allowed — `chmod`/`utimens` on a synthetic file has no
/// content effect, so it fails closed rather than silently pretending to succeed.
fn deny_mutate_path(path: *const c_char, pass: impl FnOnce() -> c_int) -> c_int {
    let Some(_scope) = HookScope::enter() else {
        return pass();
    };
    // SAFETY: caller's NUL-terminated path (or null).
    let Some(decoded) = (unsafe { decode_cstr(path) }) else {
        return pass();
    };
    match route(&decoded) {
        None | Some((_, Route::Real)) => pass(),
        Some((_, Route::Rejected | Route::Virtual(_))) => {
            set_errno(libc::EROFS);
            -1
        }
    }
}

/// Translate a VFS mutation result into the POSIX `0`/`-1` + `errno` convention.
fn mutation_result_errno(res: io::Result<()>) -> c_int {
    match res {
        Ok(()) => 0,
        Err(err) => {
            set_errno(io_to_errno(&err));
            -1
        }
    }
}

/// A path-routed **content** mutation (remove/mkdir/truncate): perform `op` against
/// the VFS for a virtual path when writes are permitted, deny (`EROFS`) when they
/// are not, reject an unsafe path (`ENOENT`), and passthrough a real path.
fn mutate_path(
    path: *const c_char,
    op: impl FnOnce(&Engine, &Path) -> Option<io::Result<()>>,
    pass: impl FnOnce() -> c_int,
) -> c_int {
    let Some(_scope) = HookScope::enter() else {
        return pass();
    };
    // SAFETY: caller's NUL-terminated path (or null).
    let Some(decoded) = (unsafe { decode_cstr(path) }) else {
        return pass();
    };
    match route(&decoded) {
        None | Some((_, Route::Real)) => pass(),
        Some((_, Route::Rejected)) => {
            set_errno(libc::ENOENT);
            -1
        }
        Some((engine, Route::Virtual(p))) => {
            if !engine.allow_writes() {
                set_errno(libc::EROFS);
                return -1;
            }
            if let Some(res) = op(&engine, &p) {
                mutation_result_errno(res)
            } else {
                set_errno(libc::ENOENT); // VFS declined a virtual path: fail closed
                -1
            }
        }
    }
}

/// `truncate` shim — path-based set-length of a virtual file when writes are
/// permitted; `EROFS`/`ENOENT`/`EINVAL` otherwise; passthrough for a real path
/// (where a negative length reaches the kernel, which returns `EINVAL`).
pub(crate) extern "C" fn shim_truncate(path: *const c_char, length: off_t) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        mutate_path(
            path,
            |engine, p| {
                if length < 0 {
                    return Some(Err(io::Error::from_raw_os_error(libc::EINVAL)));
                }
                engine.vfs().set_len(p, length as u64)
            },
            || {
                let orig: PfnTruncate = orig_fn(Sym::Truncate);
                // SAFETY: forwarding the caller's exact arguments.
                unsafe { orig(path, length) }
            },
        )
    })
}

/// `mkdir` shim — create a virtual directory when writes are permitted; fail closed
/// otherwise; passthrough for a real path.
pub(crate) extern "C" fn shim_mkdir(path: *const c_char, mode: mode_t) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        mutate_path(
            path,
            |engine, p| engine.vfs().create_dir(p),
            || {
                let orig: PfnMkdir = orig_fn(Sym::Mkdir);
                // SAFETY: forwarding the caller's exact arguments.
                unsafe { orig(path, mode) }
            },
        )
    })
}

/// `remove` shim — remove a virtual file/dir when writes are permitted; fail closed
/// otherwise; passthrough for a real path.
pub(crate) extern "C" fn shim_remove(path: *const c_char) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        mutate_path(
            path,
            |engine, p| engine.vfs().remove(p),
            || {
                let orig: PfnPathOnly = orig_fn(Sym::Remove);
                // SAFETY: forwarding the caller's exact argument.
                unsafe { orig(path) }
            },
        )
    })
}

/// `unlink` shim — remove a virtual file when writes are permitted; fail closed
/// otherwise; passthrough for a real path.
pub(crate) extern "C" fn shim_unlink(path: *const c_char) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        mutate_path(
            path,
            |engine, p| engine.vfs().remove(p),
            || {
                let orig: PfnPathOnly = orig_fn(Sym::Unlink);
                // SAFETY: forwarding the caller's exact argument.
                unsafe { orig(path) }
            },
        )
    })
}

/// A two-path mutation (`rename`/`link`/`symlink`) denied if either side is
/// virtual.
fn deny_mutate_two(from: *const c_char, to: *const c_char, pass: impl FnOnce() -> c_int) -> c_int {
    let Some(_scope) = HookScope::enter() else {
        return pass();
    };
    // SAFETY: caller's NUL-terminated paths (or null).
    let a = unsafe { decode_cstr(from) };
    let b = unsafe { decode_cstr(to) };
    let touches_virtual = [a, b]
        .into_iter()
        .flatten()
        .any(|p| matches!(route(&p), Some((_, Route::Virtual(_) | Route::Rejected))));
    if touches_virtual {
        set_errno(libc::EROFS);
        -1
    } else {
        pass()
    }
}

/// `rename` shim — move within the virtual tree (a temp-then-rename finalization)
/// when both endpoints are virtual and writes are permitted; deny a virtual↔real
/// crossing (fail closed); passthrough two real paths.
pub(crate) extern "C" fn shim_rename(from: *const c_char, to: *const c_char) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnRename = orig_fn(Sym::Rename);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(from, to) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated source path (or null).
        let Some(from_s) = (unsafe { decode_cstr(from) }) else {
            return pass();
        };
        match route(&from_s) {
            None | Some((_, Route::Real)) => {
                // Real source: deny only if the destination is virtual (a real→virtual
                // move would smuggle a disk file into the synthetic tree).
                // SAFETY: caller's NUL-terminated destination path (or null).
                let to_virtual = unsafe { decode_cstr(to) }.is_some_and(|t| {
                    matches!(route(&t), Some((_, Route::Virtual(_) | Route::Rejected)))
                });
                if to_virtual {
                    set_errno(libc::EROFS);
                    -1
                } else {
                    pass()
                }
            }
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                -1
            }
            Some((engine, Route::Virtual(f))) => {
                if !engine.allow_writes() {
                    set_errno(libc::EROFS);
                    return -1;
                }
                // SAFETY: caller's NUL-terminated destination path (or null).
                let Some(to_s) = (unsafe { decode_cstr(to) }) else {
                    set_errno(libc::EINVAL);
                    return -1;
                };
                if let Some((_, Route::Virtual(t))) = route(&to_s) {
                    if let Some(res) = engine.vfs().rename(&f, &t) {
                        mutation_result_errno(res)
                    } else {
                        set_errno(libc::ENOENT);
                        -1
                    }
                } else {
                    // A virtual→real move would leak the synthetic file to disk.
                    set_errno(libc::EROFS);
                    -1
                }
            }
        }
    })
}

/// `link` shim.
pub(crate) extern "C" fn shim_link(from: *const c_char, to: *const c_char) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        deny_mutate_two(from, to, || {
            let orig: PfnRename = orig_fn(Sym::Link);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(from, to) }
        })
    })
}

/// `symlink` shim (`target`, `linkpath`).
pub(crate) extern "C" fn shim_symlink(target: *const c_char, linkpath: *const c_char) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        deny_mutate_two(target, linkpath, || {
            let orig: PfnRename = orig_fn(Sym::Symlink);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(target, linkpath) }
        })
    })
}

/// `sendfile` shim — a virtual source fd must not silently bypass the VFS; fail
/// closed (`EINVAL`) in the read-only milestone (R17). Passthrough otherwise. Linux
/// only — macOS `sendfile` has a different (header/trailer) ABI and is unused here.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_sendfile(
    out_fd: c_int,
    in_fd: c_int,
    offset: *mut off_t,
    count: size_t,
) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnSendfile = orig_fn(Sym::Sendfile);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(out_fd, in_fd, offset, count) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().get_file(in_fd as usize).is_some() {
            set_errno(libc::EINVAL); // do not silently bypass the VFS
            -1
        } else {
            pass()
        }
    })
}

/// `fchmod` shim — an attribute mutation on a virtual fd is denied (`EROFS`).
pub(crate) extern "C" fn shim_fchmod(fd: c_int, mode: mode_t) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFchmod = orig_fn(Sym::Fchmod);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, mode) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if deny_write_fd(&engine, fd) {
            set_errno(libc::EROFS);
            -1
        } else {
            pass()
        }
    })
}

/// `fchmodat` shim — path-routed attribute mutation, denied (`EROFS`) for virtual.
pub(crate) extern "C" fn shim_fchmodat(
    dirfd: c_int,
    path: *const c_char,
    mode: mode_t,
    flags: c_int,
) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        deny_mutate_path(path, || {
            let orig: PfnFchmodat = orig_fn(Sym::Fchmodat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(dirfd, path, mode, flags) }
        })
    })
}

/// `utimensat` shim — path-routed timestamp mutation, denied (`EROFS`) for virtual.
pub(crate) extern "C" fn shim_utimensat(
    dirfd: c_int,
    path: *const c_char,
    times: *const libc::timespec,
    flags: c_int,
) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        deny_mutate_path(path, || {
            let orig: PfnUtimensat = orig_fn(Sym::Utimensat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(dirfd, path, times, flags) }
        })
    })
}

// ---- Stat family -----------------------------------------------------------

/// Metadata for a virtual path, or `None` if the VFS does not have it.
fn virtual_meta(engine: &Engine, path: &Path) -> Option<VfsMetadata> {
    match engine.vfs().metadata(path) {
        Some(Ok(meta)) => Some(meta),
        Some(Err(_)) | None => None,
    }
}

/// The virtual open behind a registry fd, if any.
fn virtual_fd_meta(engine: &Engine, fd: c_int) -> Option<(std::path::PathBuf, u64)> {
    let file = engine.registry().get_file(fd as usize)?;
    let mut open = lock(&file);
    let size = open.stream.size().unwrap_or(0);
    Some((open.path.clone(), size))
}

/// `__xstat` (versioned) shim — glibc's `stat` ABI wrapper (Linux only).
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_xstat(ver: c_int, path: *const c_char, buf: *mut stat) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnXstat = orig_fn(Sym::Xstat);
            // SAFETY: forwarding the caller's exact arguments (honoring `__ver`).
            unsafe { orig(ver, path, buf) }
        };
        stat_path(path, buf, pass, fill_stat)
    })
}

/// `__lxstat` (versioned lstat) shim — a virtual path is never a symlink, so it
/// stats identically to `__xstat`.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_lxstat(ver: c_int, path: *const c_char, buf: *mut stat) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnXstat = orig_fn(Sym::Lxstat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(ver, path, buf) }
        };
        stat_path(path, buf, pass, fill_stat)
    })
}

/// `__xstat64` (versioned) shim.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_xstat64(ver: c_int, path: *const c_char, buf: *mut stat64) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnXstat64 = orig_fn(Sym::Xstat64);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(ver, path, buf) }
        };
        stat_path(path, buf, pass, fill_stat64)
    })
}

/// `__lxstat64` (versioned lstat) shim.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_lxstat64(ver: c_int, path: *const c_char, buf: *mut stat64) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnXstat64 = orig_fn(Sym::Lxstat64);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(ver, path, buf) }
        };
        stat_path(path, buf, pass, fill_stat64)
    })
}

/// `stat` shim (glibc �A 2.33 direct export).
pub(crate) extern "C" fn shim_stat(path: *const c_char, buf: *mut stat) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnStat = orig_fn(Sym::Stat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, buf) }
        };
        stat_path(path, buf, pass, fill_stat)
    })
}

/// `lstat` shim.
pub(crate) extern "C" fn shim_lstat(path: *const c_char, buf: *mut stat) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnStat = orig_fn(Sym::Lstat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, buf) }
        };
        stat_path(path, buf, pass, fill_stat)
    })
}

/// `fstatat` shim (path-routed; absolute virtual paths ignore `dirfd`).
pub(crate) extern "C" fn shim_fstatat(
    dirfd: c_int,
    path: *const c_char,
    buf: *mut stat,
    flags: c_int,
) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFstatat = orig_fn(Sym::Fstatat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(dirfd, path, buf, flags) }
        };
        stat_path(path, buf, pass, fill_stat)
    })
}

/// `statx` shim (path-routed) — Linux only.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_statx(
    dirfd: c_int,
    path: *const c_char,
    flags: c_int,
    mask: libc::c_uint,
    buf: *mut statx,
) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnStatx = orig_fn(Sym::Statx);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(dirfd, path, flags, mask, buf) }
        };
        stat_path(path, buf, pass, fill_statx)
    })
}

/// Shared path-routed stat: fill `buf` from the VFS if `path` is virtual, else
/// passthrough. `fill` writes the platform stat struct.
fn stat_path<S>(
    path: *const c_char,
    buf: *mut S,
    pass: impl FnOnce() -> c_int,
    fill: unsafe fn(*mut S, &Path, &VfsMetadata),
) -> c_int {
    let Some(_scope) = HookScope::enter() else {
        return pass();
    };
    // SAFETY: caller's NUL-terminated path (or null).
    let Some(decoded) = (unsafe { decode_cstr(path) }) else {
        return pass();
    };
    match route(&decoded) {
        None | Some((_, Route::Real)) => pass(),
        Some((_, Route::Rejected)) => {
            set_errno(libc::ENOENT);
            -1
        }
        Some((engine, Route::Virtual(p))) => {
            if let Some(meta) = virtual_meta(&engine, &p) {
                // SAFETY: `buf` is the caller's writable stat struct.
                unsafe { fill(buf, &p, &meta) };
                0
            } else {
                set_errno(libc::ENOENT);
                -1
            }
        }
    }
}

/// `__fxstat` (versioned fstat) shim — registry-routed by fd (Linux only).
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_fxstat(ver: c_int, fd: c_int, buf: *mut stat) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFxstat = orig_fn(Sym::Fxstat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(ver, fd, buf) }
        };
        fstat_fd(fd, buf, pass, fill_stat)
    })
}

/// `__fxstat64` shim.
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_fxstat64(ver: c_int, fd: c_int, buf: *mut stat64) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFxstat64 = orig_fn(Sym::Fxstat64);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(ver, fd, buf) }
        };
        fstat_fd(fd, buf, pass, fill_stat64)
    })
}

/// `fstat` shim.
pub(crate) extern "C" fn shim_fstat(fd: c_int, buf: *mut stat) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFstat = orig_fn(Sym::Fstat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, buf) }
        };
        fstat_fd(fd, buf, pass, fill_stat)
    })
}

/// Shared fd-routed fstat.
fn fstat_fd<S>(
    fd: c_int,
    buf: *mut S,
    pass: impl FnOnce() -> c_int,
    fill: unsafe fn(*mut S, &Path, &VfsMetadata),
) -> c_int {
    let Some(_scope) = HookScope::enter() else {
        return pass();
    };
    let Some(engine) = current_engine() else {
        return pass();
    };
    match virtual_fd_meta(&engine, fd) {
        Some((path, size)) => {
            let meta = VfsMetadata::file(size);
            // SAFETY: `buf` is the caller's writable stat struct.
            unsafe { fill(buf, &path, &meta) };
            0
        }
        None => pass(),
    }
}

// ---- Volume ----------------------------------------------------------------

/// Fill a `struct statvfs` from the synthetic volume geometry (the provider's
/// configurable capacity when it reports one, else a roomy default).
///
/// # Safety
/// `buf` must be a valid, writable `*mut statvfs`.
unsafe fn fill_statvfs(buf: *mut statvfs, info: Option<VolumeInfo>) {
    if buf.is_null() {
        return;
    }
    // SAFETY: `statvfs` is plain-old-data; all-zero is a valid bit pattern.
    let mut vfs: statvfs = unsafe { core::mem::zeroed() };
    // Block counts are `fsblkcnt_t` — `c_ulong` (u64) on glibc, `c_uint` (u32) on
    // macOS — so every count saturates into the target's own type. `try_from` is
    // genuinely fallible on macOS (u64 -> u32) but infallible on glibc (u64 -> u64),
    // hence the Linux-only allow, matching `st_dev` in `fill_stat`.
    let (bsize, total, free) = match info {
        Some(v) => {
            let bsize = u64::from(v.block_size.max(512));
            (bsize, v.capacity / bsize, v.available / bsize)
        }
        // A roomy default: `1 << 32` total 4 KiB blocks with half free.
        None => (4096, 1u64 << 32, 1u64 << 31),
    };
    #[cfg_attr(target_os = "linux", allow(clippy::unnecessary_fallible_conversions))]
    {
        vfs.f_bsize = libc::c_ulong::try_from(bsize).unwrap_or(4096);
        vfs.f_frsize = vfs.f_bsize;
        vfs.f_blocks = libc::fsblkcnt_t::try_from(total).unwrap_or(libc::fsblkcnt_t::MAX);
        vfs.f_bfree = libc::fsblkcnt_t::try_from(free).unwrap_or(libc::fsblkcnt_t::MAX);
        vfs.f_bavail = vfs.f_bfree;
    }
    vfs.f_files = 1 << 20;
    vfs.f_ffree = 1 << 19;
    vfs.f_favail = 1 << 19;
    vfs.f_fsid = u64::from(VIRTUAL_VOLUME_SERIAL);
    vfs.f_namemax = 255;
    // SAFETY: caller guarantees `buf` is writable.
    unsafe { *buf = vfs };
}

/// `statvfs` shim (path-routed).
pub(crate) extern "C" fn shim_statvfs(path: *const c_char, buf: *mut statvfs) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnStatvfs = orig_fn(Sym::Statvfs);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, buf) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                -1
            }
            Some((engine, Route::Virtual(_))) => {
                // SAFETY: `buf` is the caller's writable statvfs.
                unsafe { fill_statvfs(buf, engine.vfs().volume_info()) };
                0
            }
        }
    })
}

/// `fstatvfs` shim (registry-routed by fd).
pub(crate) extern "C" fn shim_fstatvfs(fd: c_int, buf: *mut statvfs) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFstatvfs = orig_fn(Sym::Fstatvfs);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(fd, buf) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().get_file(fd as usize).is_some() {
            // SAFETY: `buf` is the caller's writable statvfs.
            unsafe { fill_statvfs(buf, engine.vfs().volume_info()) };
            0
        } else {
            pass()
        }
    })
}

// ---- Directory family ------------------------------------------------------

/// The enumeration state behind a virtual `DIR*`: the listing, a cursor, and the
/// scratch `dirent`/`dirent64` the POSIX API returns pointers into (valid until the
/// next `readdir` call). `dirent64` is glibc-only.
struct VirtualDir {
    entries: Vec<VfsDirEntry>,
    next: usize,
    dirent: dirent,
    #[cfg(target_os = "linux")]
    dirent64: dirent64,
}

/// Fill a `dirent` name field from an entry, returning a pointer to the scratch
/// struct. `d_ino`/offset/`d_type` are set to plausible values. The seek-offset
/// field differs (`d_off` on glibc, `d_seekoff` on macOS, which also carries
/// `d_namlen`), so it is cfg-split; the macOS `dirent` is the `$INODE64` layout.
fn fill_dirent(entry: &VfsDirEntry, slot: &mut dirent, index: usize) -> *mut dirent {
    // Set the identifying fields; `d_name` is a fixed C array.
    slot.d_ino = stable_ino(Path::new(&entry.name)).max(1);
    slot.d_reclen = core::mem::size_of::<dirent>() as u16;
    slot.d_type = if entry.metadata.is_dir {
        libc::DT_DIR
    } else {
        libc::DT_REG
    };
    let written = write_dname(&mut slot.d_name, &entry.name.to_string_lossy());
    #[cfg(target_os = "linux")]
    {
        slot.d_off = (index + 1) as off_t;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        slot.d_seekoff = (index + 1) as u64;
        slot.d_namlen = u16::try_from(written).unwrap_or(u16::MAX);
    }
    let _ = written; // used only on Darwin (macOS + iOS)
    core::ptr::from_mut(slot)
}

#[cfg(target_os = "linux")]
fn fill_dirent64(entry: &VfsDirEntry, slot: &mut dirent64, index: usize) -> *mut dirent64 {
    slot.d_ino = stable_ino(Path::new(&entry.name)).max(1);
    slot.d_off = (index + 1) as i64;
    slot.d_reclen = core::mem::size_of::<dirent64>() as u16;
    slot.d_type = if entry.metadata.is_dir {
        libc::DT_DIR
    } else {
        libc::DT_REG
    };
    write_dname(&mut slot.d_name, &entry.name.to_string_lossy());
    core::ptr::from_mut(slot)
}

/// Copy a name into a fixed `d_name[c_char]` array, NUL-terminated and truncated;
/// returns the number of name bytes written (excluding the NUL).
fn write_dname(dst: &mut [c_char], name: &str) -> usize {
    let bytes = name.as_bytes();
    let n = bytes.len().min(dst.len().saturating_sub(1));
    for (slot, &b) in dst.iter_mut().zip(bytes.iter()).take(n) {
        *slot = b as c_char;
    }
    if let Some(term) = dst.get_mut(n) {
        *term = 0;
    }
    n
}

/// `opendir` shim — a virtual directory becomes a `Box<VirtualDir>` handle.
pub(crate) extern "C" fn shim_opendir(path: *const c_char) -> *mut DIR {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnOpendir = orig_fn(Sym::Opendir);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(path) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                core::ptr::null_mut()
            }
            Some((engine, Route::Virtual(p))) => match engine.vfs().read_dir(&p) {
                Some(Ok(entries)) => {
                    // SAFETY: dirent scratch is POD; zeroed is valid.
                    let dir = Box::new(VirtualDir {
                        entries,
                        next: 0,
                        dirent: unsafe { core::mem::zeroed() },
                        #[cfg(target_os = "linux")]
                        dirent64: unsafe { core::mem::zeroed() },
                    });
                    let raw = Box::into_raw(dir);
                    engine.registry().insert_dir(raw as usize);
                    raw.cast::<DIR>()
                }
                Some(Err(err)) => {
                    set_errno(io_to_errno(&err));
                    core::ptr::null_mut()
                }
                None => {
                    set_errno(libc::ENOENT);
                    core::ptr::null_mut()
                }
            },
        }
    })
}

/// `fdopendir` shim — a virtual carrier fd is a regular file, not a directory, so
/// fail closed (`ENOTDIR`); passthrough for a real fd.
pub(crate) extern "C" fn shim_fdopendir(fd: c_int) -> *mut DIR {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFdopendir = orig_fn(Sym::Fdopendir);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(fd) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().get_file(fd as usize).is_some() {
            set_errno(libc::ENOTDIR);
            core::ptr::null_mut()
        } else {
            pass()
        }
    })
}

/// `readdir` shim.
pub(crate) extern "C" fn shim_readdir(dirp: *mut DIR) -> *mut dirent {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnReaddir = orig_fn(Sym::Readdir);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(dirp) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if !engine.registry().contains_dir(dirp as usize) {
            return pass();
        }
        // SAFETY: membership proves `dirp` is a `Box<VirtualDir>` we created and
        // still own; POSIX forbids concurrent `readdir` on the same `DIR*`.
        let dir = unsafe { &mut *dirp.cast::<VirtualDir>() };
        let idx = dir.next;
        let Some(entry) = dir.entries.get(idx).cloned() else {
            return core::ptr::null_mut(); // end of directory
        };
        dir.next = idx + 1;
        fill_dirent(&entry, &mut dir.dirent, idx)
    })
}

/// `readdir64` shim (Linux; the `dirent64` layout).
#[cfg(target_os = "linux")]
pub(crate) extern "C" fn shim_readdir64(dirp: *mut DIR) -> *mut dirent64 {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnReaddir64 = orig_fn(Sym::Readdir64);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(dirp) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if !engine.registry().contains_dir(dirp as usize) {
            return pass();
        }
        // SAFETY: as in `shim_readdir`.
        let dir = unsafe { &mut *dirp.cast::<VirtualDir>() };
        let idx = dir.next;
        let Some(entry) = dir.entries.get(idx).cloned() else {
            return core::ptr::null_mut();
        };
        dir.next = idx + 1;
        fill_dirent64(&entry, &mut dir.dirent64, idx)
    })
}

/// `closedir` shim — drop the virtual dir once; passthrough for a real `DIR*`.
pub(crate) extern "C" fn shim_closedir(dirp: *mut DIR) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnClosedir = orig_fn(Sym::Closedir);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(dirp) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().remove_dir(dirp as usize) {
            // SAFETY: membership proved `dirp` is our `Box<VirtualDir>`; reclaim it
            // exactly once.
            drop(unsafe { Box::from_raw(dirp.cast::<VirtualDir>()) });
            0
        } else {
            pass()
        }
    })
}

// ---- Path family -----------------------------------------------------------

/// `realpath` shim — a virtual path is already canonical: copy it back (or malloc
/// a copy when `resolved` is null), preserving the virtual prefix.
pub(crate) extern "C" fn shim_realpath(path: *const c_char, resolved: *mut c_char) -> *mut c_char {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnRealpath = orig_fn(Sym::Realpath);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, resolved) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                core::ptr::null_mut()
            }
            Some((engine, Route::Virtual(p))) => {
                // Confirm the virtual path exists before "resolving" it.
                if virtual_meta(&engine, &p).is_none() {
                    set_errno(libc::ENOENT);
                    return core::ptr::null_mut();
                }
                copy_out_path(&p.to_string_lossy(), resolved)
            }
        }
    })
}

/// Write a resolved path into the caller's `resolved` buffer (assumed `PATH_MAX`)
/// or, when null, into a freshly `malloc`ed buffer, both NUL-terminated. Returns
/// the buffer pointer, or null with `errno` on failure.
fn copy_out_path(path: &str, resolved: *mut c_char) -> *mut c_char {
    let bytes = path.as_bytes();
    let need = bytes.len() + 1;
    if resolved.is_null() {
        // SAFETY: `malloc(need)` returns `need` writable bytes or null.
        let out = unsafe { libc::malloc(need) }.cast::<c_char>();
        if out.is_null() {
            set_errno(libc::ENOMEM);
            return core::ptr::null_mut();
        }
        // SAFETY: `out` addresses `need` writable bytes.
        unsafe { write_cstr(out, need, bytes) };
        out
    } else {
        // The caller's buffer is documented to be at least `PATH_MAX` bytes.
        if need > libc::PATH_MAX as usize {
            set_errno(libc::ENAMETOOLONG);
            return core::ptr::null_mut();
        }
        // SAFETY: `resolved` addresses at least `PATH_MAX` writable bytes.
        unsafe { write_cstr(resolved, libc::PATH_MAX as usize, bytes) };
        resolved
    }
}

/// Copy `bytes` + a NUL into a C buffer of `cap` bytes (truncating defensively).
///
/// # Safety
/// `dst` must address `cap` writable bytes.
unsafe fn write_cstr(dst: *mut c_char, cap: usize, bytes: &[u8]) {
    let n = bytes.len().min(cap.saturating_sub(1));
    // SAFETY: `[dst, dst+n)` is within the `cap`-byte buffer.
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), dst, n) };
    // SAFETY: `dst+n` is within the buffer (`n < cap`).
    unsafe { *dst.add(n) = 0 };
}

/// `access` shim.
pub(crate) extern "C" fn shim_access(path: *const c_char, mode: c_int) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnAccess = orig_fn(Sym::Access);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, mode) }
        };
        access_common(path, mode, pass)
    })
}

/// `faccessat` shim (path-routed; absolute virtual paths ignore `dirfd`).
pub(crate) extern "C" fn shim_faccessat(
    dirfd: c_int,
    path: *const c_char,
    mode: c_int,
    flags: c_int,
) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnFaccessat = orig_fn(Sym::Faccessat);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(dirfd, path, mode, flags) }
        };
        access_common(path, mode, pass)
    })
}

fn access_common(path: *const c_char, mode: c_int, pass: impl FnOnce() -> c_int) -> c_int {
    let Some(_scope) = HookScope::enter() else {
        return pass();
    };
    // SAFETY: caller's NUL-terminated path (or null).
    let Some(decoded) = (unsafe { decode_cstr(path) }) else {
        return pass();
    };
    match route(&decoded) {
        None | Some((_, Route::Real)) => pass(),
        Some((_, Route::Rejected)) => {
            set_errno(libc::ENOENT);
            -1
        }
        Some((engine, Route::Virtual(p))) => {
            if engine.vfs().exists(&p) != Some(true) {
                set_errno(libc::ENOENT);
                return -1;
            }
            if mode & libc::W_OK != 0 && !engine.allow_writes() {
                set_errno(libc::EROFS); // read-only milestone
                return -1;
            }
            0
        }
    }
}

/// `getcwd` shim — return the tracked virtual cwd when set, else passthrough.
pub(crate) extern "C" fn shim_getcwd(buf: *mut c_char, size: size_t) -> *mut c_char {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnGetcwd = orig_fn(Sym::Getcwd);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(buf, size) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(cwd) = engine.virtual_cwd() else {
            return pass();
        };
        let bytes = cwd.as_bytes();
        let need = bytes.len() + 1;
        if buf.is_null() {
            // glibc extension: allocate a buffer of `size` (or the needed length).
            let cap = if size == 0 { need } else { size };
            if need > cap {
                set_errno(libc::ERANGE);
                return core::ptr::null_mut();
            }
            // SAFETY: `malloc(cap)` returns `cap` writable bytes or null.
            let out = unsafe { libc::malloc(cap) }.cast::<c_char>();
            if out.is_null() {
                set_errno(libc::ENOMEM);
                return core::ptr::null_mut();
            }
            // SAFETY: `out` addresses `cap` writable bytes.
            unsafe { write_cstr(out, cap, bytes) };
            out
        } else {
            if need > size {
                set_errno(libc::ERANGE);
                return core::ptr::null_mut();
            }
            // SAFETY: `buf` addresses `size` writable bytes.
            unsafe { write_cstr(buf, size, bytes) };
            buf
        }
    })
}

/// `chdir` shim — track a virtual cwd; clear it (and passthrough) for a real path.
pub(crate) extern "C" fn shim_chdir(path: *const c_char) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnPathOnly = orig_fn(Sym::Chdir);
            // SAFETY: forwarding the caller's exact argument.
            unsafe { orig(path) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match route(&decoded) {
            None => pass(),
            Some((engine, Route::Virtual(p))) => {
                if engine
                    .vfs()
                    .metadata(&p)
                    .is_some_and(|m| m.is_ok_and(|m| m.is_dir))
                {
                    engine.set_virtual_cwd(Some(p.to_string_lossy().into_owned()));
                    0
                } else {
                    set_errno(libc::ENOENT);
                    -1
                }
            }
            Some((engine, Route::Real)) => {
                engine.set_virtual_cwd(None); // leaving the virtual tree
                pass()
            }
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                -1
            }
        }
    })
}

/// `readlink` shim — a virtual path is never a symlink: fail closed (`EINVAL`).
pub(crate) extern "C" fn shim_readlink(
    path: *const c_char,
    buf: *mut c_char,
    bufsiz: size_t,
) -> ssize_t {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnReadlink = orig_fn(Sym::Readlink);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(path, buf, bufsiz) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Virtual(_))) => {
                set_errno(libc::EINVAL); // not a symbolic link
                -1
            }
            Some((_, Route::Rejected)) => {
                set_errno(libc::ENOENT);
                -1
            }
        }
    })
}

// ---- Loader (passthrough + auto_rescan) ------------------------------------

/// `dlopen` shim — passthrough, then re-run installation on the newly loaded
/// object (when `auto_rescan` is enabled), so late decoder plugins are patched.
pub(crate) extern "C" fn shim_dlopen(path: *const c_char, flags: c_int) -> *mut c_void {
    guard_abi(core::ptr::null_mut(), PANIC_ERRNO, || {
        let orig: PfnDlopen = orig_fn(Sym::Dlopen);
        // SAFETY: forwarding the caller's exact arguments.
        let handle = unsafe { orig(path, flags) };
        if !handle.is_null() {
            dispatch::trigger_rescan();
        }
        handle
    })
}

/// `dlclose` shim — pure passthrough (recorded so it is not "unexamined").
pub(crate) extern "C" fn shim_dlclose(handle: *mut c_void) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let orig: PfnDlclose = orig_fn(Sym::Dlclose);
        // SAFETY: forwarding the caller's exact argument.
        unsafe { orig(handle) }
    })
}

// ---- Install support -------------------------------------------------------

/// The replacement function pointer for `sym` — the trampoline for the variadic
/// open family, the plain shim otherwise (Linux/glibc symbol set). macOS has its
/// own mapping in [`macos::shim_address`], sharing the portable shims defined here.
#[cfg(target_os = "linux")]
pub(crate) fn shim_address(sym: Sym) -> *const c_void {
    match sym {
        Sym::Open | Sym::Open64 | Sym::Openat => trampolines::trampoline_address(sym),
        Sym::Fopen => shim_fopen as *const c_void,
        Sym::Fopen64 => shim_fopen64 as *const c_void,
        Sym::Read => shim_read as *const c_void,
        Sym::Readv => shim_readv as *const c_void,
        Sym::Pread => shim_pread as *const c_void,
        Sym::Lseek => shim_lseek as *const c_void,
        Sym::Lseek64 => shim_lseek64 as *const c_void,
        Sym::Close => shim_close as *const c_void,
        Sym::Write => shim_write as *const c_void,
        Sym::Writev => shim_writev as *const c_void,
        Sym::Pwrite => shim_pwrite as *const c_void,
        Sym::Ftruncate => shim_ftruncate as *const c_void,
        Sym::Ftruncate64 => shim_ftruncate64 as *const c_void,
        Sym::Truncate => shim_truncate as *const c_void,
        Sym::Fsync => shim_fsync as *const c_void,
        Sym::Mkdir => shim_mkdir as *const c_void,
        Sym::Remove => shim_remove as *const c_void,
        Sym::Rename => shim_rename as *const c_void,
        Sym::Unlink => shim_unlink as *const c_void,
        Sym::Link => shim_link as *const c_void,
        Sym::Symlink => shim_symlink as *const c_void,
        Sym::Sendfile => shim_sendfile as *const c_void,
        Sym::Fchmod => shim_fchmod as *const c_void,
        Sym::Fchmodat => shim_fchmodat as *const c_void,
        Sym::Utimensat => shim_utimensat as *const c_void,
        Sym::Xstat => shim_xstat as *const c_void,
        Sym::Fxstat => shim_fxstat as *const c_void,
        Sym::Lxstat => shim_lxstat as *const c_void,
        Sym::Xstat64 => shim_xstat64 as *const c_void,
        Sym::Fxstat64 => shim_fxstat64 as *const c_void,
        Sym::Lxstat64 => shim_lxstat64 as *const c_void,
        Sym::Stat => shim_stat as *const c_void,
        Sym::Fstat => shim_fstat as *const c_void,
        Sym::Lstat => shim_lstat as *const c_void,
        Sym::Fstatat => shim_fstatat as *const c_void,
        Sym::Statx => shim_statx as *const c_void,
        Sym::Statvfs => shim_statvfs as *const c_void,
        Sym::Fstatvfs => shim_fstatvfs as *const c_void,
        Sym::Opendir => shim_opendir as *const c_void,
        Sym::Fdopendir => shim_fdopendir as *const c_void,
        Sym::Readdir => shim_readdir as *const c_void,
        Sym::Readdir64 => shim_readdir64 as *const c_void,
        Sym::Closedir => shim_closedir as *const c_void,
        Sym::Realpath => shim_realpath as *const c_void,
        Sym::Access => shim_access as *const c_void,
        Sym::Faccessat => shim_faccessat as *const c_void,
        Sym::Getcwd => shim_getcwd as *const c_void,
        Sym::Chdir => shim_chdir as *const c_void,
        Sym::Readlink => shim_readlink as *const c_void,
        Sym::Dlopen => shim_dlopen as *const c_void,
        Sym::Dlclose => shim_dlclose as *const c_void,
    }
}

/// The replacement function pointer for `sym` (Darwin symbol set — macOS + iOS).
/// Reuses the portable shims above; the Darwin-specific `fopen` (via [`shim_fopen`]'s
/// [`fopen_virtual_dispatch`]) and `__getdirentries64` come from the [`macos`]
/// submodule. There are no `*64` / `__?xstat` / `statx` / `sendfile` entries.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) fn shim_address(sym: Sym) -> *const c_void {
    match sym {
        Sym::Open | Sym::Openat => trampolines::trampoline_address(sym),
        Sym::Fopen => shim_fopen as *const c_void,
        Sym::Read => shim_read as *const c_void,
        Sym::Readv => shim_readv as *const c_void,
        Sym::Pread => shim_pread as *const c_void,
        Sym::Lseek => shim_lseek as *const c_void,
        Sym::Close => shim_close as *const c_void,
        Sym::Write => shim_write as *const c_void,
        Sym::Writev => shim_writev as *const c_void,
        Sym::Pwrite => shim_pwrite as *const c_void,
        Sym::Ftruncate => shim_ftruncate as *const c_void,
        Sym::Truncate => shim_truncate as *const c_void,
        Sym::Fsync => shim_fsync as *const c_void,
        Sym::Mkdir => shim_mkdir as *const c_void,
        Sym::Remove => shim_remove as *const c_void,
        Sym::Rename => shim_rename as *const c_void,
        Sym::Unlink => shim_unlink as *const c_void,
        Sym::Link => shim_link as *const c_void,
        Sym::Symlink => shim_symlink as *const c_void,
        Sym::Fchmod => shim_fchmod as *const c_void,
        Sym::Fchmodat => shim_fchmodat as *const c_void,
        Sym::Utimensat => shim_utimensat as *const c_void,
        Sym::Stat => shim_stat as *const c_void,
        Sym::Fstat => shim_fstat as *const c_void,
        Sym::Lstat => shim_lstat as *const c_void,
        Sym::Fstatat => shim_fstatat as *const c_void,
        Sym::Statvfs => shim_statvfs as *const c_void,
        Sym::Fstatvfs => shim_fstatvfs as *const c_void,
        Sym::Opendir => shim_opendir as *const c_void,
        Sym::Fdopendir => shim_fdopendir as *const c_void,
        Sym::Readdir => shim_readdir as *const c_void,
        Sym::Closedir => shim_closedir as *const c_void,
        Sym::Getdirentries64 => macos::shim_getdirentries64 as *const c_void,
        Sym::Realpath => shim_realpath as *const c_void,
        Sym::Access => shim_access as *const c_void,
        Sym::Faccessat => shim_faccessat as *const c_void,
        Sym::Getcwd => shim_getcwd as *const c_void,
        Sym::Chdir => shim_chdir as *const c_void,
        Sym::Readlink => shim_readlink as *const c_void,
        Sym::Dlopen => shim_dlopen as *const c_void,
        Sym::Dlclose => shim_dlclose as *const c_void,
    }
}

// ---- stat / dirent ABI layout assertions (R13) -----------------------------
//
// The shims write directly into the caller's `struct stat`/`stat64`/`statx`/
// `dirent` via the `libc` structs, so the layout is glibc's by construction. These
// tests exercise the fill functions to pin the *values* they synthesize and prove
// the writes stay in-bounds. They run on native Linux (CI); on this Windows host
// they are cfg-compiled only.
#[cfg(all(test, target_os = "linux"))]
#[allow(
    clippy::unwrap_used,
    clippy::undocumented_unsafe_blocks,
    clippy::useless_conversion
)]
mod layout_tests {
    use super::*;

    #[test]
    fn fill_stat_reports_a_read_only_regular_file() {
        let meta = VfsMetadata::file(12_345);
        let mut st: stat = unsafe { core::mem::zeroed() };
        unsafe { fill_stat(&raw mut st, Path::new("/__hookfs__/x/clip.braw"), &meta) };
        assert_eq!(st.st_size, 12_345);
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFREG);
        assert_eq!(st.st_mode & 0o777, 0o444);
        assert_eq!(st.st_nlink, 1);
        assert_ne!(st.st_ino, 0, "a stable non-zero inode identity");
        assert_eq!(st.st_blksize, 512);
        assert_eq!(st.st_blocks, (12_345 + 511) / 512);
    }

    #[test]
    fn fill_stat64_and_statx_agree_on_size() {
        let meta = VfsMetadata::file(1 << 40); // > 4 GiB: exercises 64-bit fields
        let mut st64: stat64 = unsafe { core::mem::zeroed() };
        unsafe { fill_stat64(&raw mut st64, Path::new("/x"), &meta) };
        assert_eq!(st64.st_size, 1 << 40);
        assert_eq!(st64.st_mode & libc::S_IFMT, libc::S_IFREG);

        let mut stx: statx = unsafe { core::mem::zeroed() };
        unsafe { fill_statx(&raw mut stx, Path::new("/x"), &meta) };
        assert_eq!(stx.stx_size, 1 << 40);
        assert_ne!(stx.stx_mask & libc::STATX_BASIC_STATS, 0);
    }

    #[test]
    fn fill_stat_for_a_directory_sets_ifdir() {
        let meta = VfsMetadata::dir();
        let mut st: stat = unsafe { core::mem::zeroed() };
        unsafe { fill_stat(&raw mut st, Path::new("/d"), &meta) };
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFDIR);
    }

    #[test]
    fn dirent_name_is_nul_terminated_and_truncated() {
        let entry = VfsDirEntry {
            name: std::ffi::OsString::from("sample.braw"),
            metadata: VfsMetadata::file(1),
        };
        let mut d: dirent = unsafe { core::mem::zeroed() };
        let ptr = fill_dirent(&entry, &mut d, 0);
        assert_eq!(ptr, &raw mut d);
        assert_eq!(d.d_type, libc::DT_REG);
        // Read back the name as a C string.
        let name = unsafe { core::ffi::CStr::from_ptr(d.d_name.as_ptr()) };
        assert_eq!(name.to_str().unwrap(), "sample.braw");

        // An over-long name is truncated and still NUL-terminated (no overflow).
        let long = "a".repeat(1024);
        let entry = VfsDirEntry {
            name: std::ffi::OsString::from(&long),
            metadata: VfsMetadata::file(1),
        };
        let mut d: dirent64 = unsafe { core::mem::zeroed() };
        fill_dirent64(&entry, &mut d, 1);
        let name = unsafe { core::ffi::CStr::from_ptr(d.d_name.as_ptr()) };
        assert!(name.to_bytes().len() < d.d_name.len());
        assert!(name.to_bytes().iter().all(|&b| b == b'a'));
    }

    #[test]
    fn statvfs_reports_roomy_free_space() {
        let mut v: statvfs = unsafe { core::mem::zeroed() };
        unsafe { fill_statvfs(&raw mut v, None) };
        assert!(v.f_bavail > 0);
        assert_eq!(v.f_bsize, 4096);
        assert_eq!(v.f_namemax, 255);
    }

    #[test]
    fn statvfs_reflects_configured_capacity() {
        let info = VolumeInfo {
            capacity: 1 << 20,
            available: 1 << 19,
            block_size: 4096,
        };
        let mut v: statvfs = unsafe { core::mem::zeroed() };
        unsafe { fill_statvfs(&raw mut v, Some(info)) };
        assert_eq!(v.f_bsize, 4096);
        assert_eq!(u64::from(v.f_blocks), (1u64 << 20) / 4096);
        assert_eq!(u64::from(v.f_bavail), (1u64 << 19) / 4096);
    }
}
