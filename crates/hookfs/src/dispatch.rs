//!
//! * the **`Originals`** table — the saved real function pointers, resolved once
//!   at install *before any hook can run*, so passthrough never discovers an
//!   original recursively;
//! * the **active engine** slot — the routing state (VFS + namespace + registry +
//!   behind an `RwLock` and cloned out (an `Arc`) for the duration of each call so
//!   it is never freed mid-hook. A second overlapping install fails loudly rather
//!   than clobbering it, and only the owning guard may tear it down;
//! * the thread-local **reentrancy guard** — a re-entered shim passes straight
//!   through, so a backing stream that itself performs real I/O cannot loop;
//! * the **panic firewall** — every shim runs its Rust work inside `catch_unwind`
//!   so a panic becomes a native error instead of unwinding across the FFI
//!   boundary (R15).
//!
//! The whole dispatch engine is only meaningful where a `plthook` backend can
//! install the shims (Windows/PE, Linux/ELF, and the Darwin/Mach-O backend for
//! macOS + iOS),
//! so it is gated to the same targets as `crate::install`'s real path. The POSIX
//! `Sym`/`ORIGINALS` table is defined per-target (a glibc symbol set on Linux, a
//! shared libc symbol set on Darwin/macOS+iOS), and the `guard_abi` `errno` write
//! goes through each platform's thread-local accessor (glibc `__errno_location` on
//! Linux, `__error` on Darwin). The remaining unsupported unix targets (BSD,
//! Android/bionic) build the
//! `UnsupportedPlatform` placeholder and never reference this module.
//!
//! Gated on `hookfs_backend` — the single source of truth (see `build.rs`) for
//! `"ios"` to that list, with no edit here.
#![cfg(hookfs_backend)]

use crate::router::Engine;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, RwLock};

/// Every function `hookfs` may hook or call an original of. The discriminant is
/// the index into the [`ORIGINALS`] table. The symbol set is platform-specific
/// (KERNEL32 on Windows, libc on POSIX); the surrounding dispatch machinery — the
/// originals table, the active-engine slot, the reentrancy guard, and the panic
/// firewall — is shared.
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Sym {
    CreateFileW,
    CreateFileA,
    CreateFile2,
    ReadFile,
    ReadFileEx,
    WriteFile,
    WriteFileEx,
    SetFilePointer,
    SetFilePointerEx,
    GetFileSize,
    GetFileSizeEx,
    GetFileType,
    GetFileInformationByHandle,
    GetFileInformationByHandleEx,
    FlushFileBuffers,
    SetEndOfFile,
    CloseHandle,
    FindFirstFileExW,
    FindFirstFileW,
    FindNextFileW,
    FindClose,
    GetFileAttributesW,
    GetFileAttributesExW,
    GetFullPathNameW,
    GetFullPathNameA,
    GetCurrentDirectoryW,
    DeleteFileW,
    MoveFileExW,
    GetDiskFreeSpaceA,
    GetDriveTypeW,
    CreateDirectoryW,
    LoadLibraryExW,
    LoadLibraryA,
}

#[cfg(windows)]
impl Sym {
    /// The exported symbol name, exactly as it appears in the import table.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::CreateFileW => "CreateFileW",
            Self::CreateFileA => "CreateFileA",
            Self::CreateFile2 => "CreateFile2",
            Self::ReadFile => "ReadFile",
            Self::ReadFileEx => "ReadFileEx",
            Self::WriteFile => "WriteFile",
            Self::WriteFileEx => "WriteFileEx",
            Self::SetFilePointer => "SetFilePointer",
            Self::SetFilePointerEx => "SetFilePointerEx",
            Self::GetFileSize => "GetFileSize",
            Self::GetFileSizeEx => "GetFileSizeEx",
            Self::GetFileType => "GetFileType",
            Self::GetFileInformationByHandle => "GetFileInformationByHandle",
            Self::GetFileInformationByHandleEx => "GetFileInformationByHandleEx",
            Self::FlushFileBuffers => "FlushFileBuffers",
            Self::SetEndOfFile => "SetEndOfFile",
            Self::CloseHandle => "CloseHandle",
            Self::FindFirstFileExW => "FindFirstFileExW",
            Self::FindFirstFileW => "FindFirstFileW",
            Self::FindNextFileW => "FindNextFileW",
            Self::FindClose => "FindClose",
            Self::GetFileAttributesW => "GetFileAttributesW",
            Self::GetFileAttributesExW => "GetFileAttributesExW",
            Self::GetFullPathNameW => "GetFullPathNameW",
            Self::GetFullPathNameA => "GetFullPathNameA",
            Self::GetCurrentDirectoryW => "GetCurrentDirectoryW",
            Self::DeleteFileW => "DeleteFileW",
            Self::MoveFileExW => "MoveFileExW",
            Self::GetDiskFreeSpaceA => "GetDiskFreeSpaceA",
            Self::GetDriveTypeW => "GetDriveTypeW",
            Self::CreateDirectoryW => "CreateDirectoryW",
            Self::LoadLibraryExW => "LoadLibraryExW",
            Self::LoadLibraryA => "LoadLibraryA",
        }
    }

    /// Resolve a symbol name to its [`Sym`], if `hookfs` knows it.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        ALL.iter().copied().find(|s| s.name() == name)
    }
}

/// Every symbol, in discriminant order (also the install candidate list).
#[cfg(windows)]
pub(crate) const ALL: [Sym; Sym::COUNT] = [
    Sym::CreateFileW,
    Sym::CreateFileA,
    Sym::CreateFile2,
    Sym::ReadFile,
    Sym::ReadFileEx,
    Sym::WriteFile,
    Sym::WriteFileEx,
    Sym::SetFilePointer,
    Sym::SetFilePointerEx,
    Sym::GetFileSize,
    Sym::GetFileSizeEx,
    Sym::GetFileType,
    Sym::GetFileInformationByHandle,
    Sym::GetFileInformationByHandleEx,
    Sym::FlushFileBuffers,
    Sym::SetEndOfFile,
    Sym::CloseHandle,
    Sym::FindFirstFileExW,
    Sym::FindFirstFileW,
    Sym::FindNextFileW,
    Sym::FindClose,
    Sym::GetFileAttributesW,
    Sym::GetFileAttributesExW,
    Sym::GetFullPathNameW,
    Sym::GetFullPathNameA,
    Sym::GetCurrentDirectoryW,
    Sym::DeleteFileW,
    Sym::MoveFileExW,
    Sym::GetDiskFreeSpaceA,
    Sym::GetDriveTypeW,
    Sym::CreateDirectoryW,
    Sym::LoadLibraryExW,
    Sym::LoadLibraryA,
];

#[cfg(windows)]
impl Sym {
    /// Total number of known symbols (the size of the [`ORIGINALS`] table).
    pub(crate) const COUNT: usize = 33;
}

// ---- Linux / glibc symbol table --------------------------------------------

/// entries of `audit/allowlist.json`). The install layer applies these
/// *optionally*, so only the ones a target module actually imports are patched —
/// the shipped `libBlackmagicRawAPI.so` imports the open/read/seek/close/stat/
/// path/mutate family plus the loader hooks; the dir family bottoms out in
/// `libc++.so.1`.
///
/// The intentional **cookie-stream passthroughs** (`fread`/`fseek`/`fclose`/…) are
/// deliberately *absent*: after `fopen` returns a real `fopencookie` `FILE*`, libc
///
/// macOS has its own [`Sym`](#macos) below (no `__?xstat` / `*64` variants; the
/// x86-64 `$INODE64` stat/dir aliases; `__getdirentries64`).
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Sym {
    // Open family (the routing decision is by path).
    Fopen,
    Fopen64,
    Open,
    Open64,
    Openat,
    // Descriptor family (routing by registry membership).
    Read,
    Readv,
    Pread,
    Lseek,
    Lseek64,
    Close,
    // Write / truncate / mutate — fail closed (EROFS) for virtual targets.
    Write,
    Writev,
    Pwrite,
    Ftruncate,
    Ftruncate64,
    Truncate,
    Fsync,
    Mkdir,
    Remove,
    Rename,
    Unlink,
    Link,
    Symlink,
    Sendfile,
    // Attribute mutation (bottoms out in libc++/libc++abi) — fail closed (EROFS).
    Fchmod,
    Fchmodat,
    Utimensat,
    // Stat family (honoring the leading `__ver` of the `__?xstat` wrappers).
    Xstat,
    Fxstat,
    Lxstat,
    Xstat64,
    Fxstat64,
    Lxstat64,
    Stat,
    Fstat,
    Lstat,
    Fstatat,
    Statx,
    // Volume.
    Statvfs,
    Fstatvfs,
    // Directory family.
    Opendir,
    Fdopendir,
    Readdir,
    Readdir64,
    Closedir,
    // Path.
    Realpath,
    Access,
    Faccessat,
    Getcwd,
    Chdir,
    Readlink,
    // Loader (passthrough + auto_rescan).
    Dlopen,
    Dlclose,
}

#[cfg(target_os = "linux")]
impl Sym {
    /// The exported symbol name, exactly as it appears in the dynamic symbol table.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Fopen => "fopen",
            Self::Fopen64 => "fopen64",
            Self::Open => "open",
            Self::Open64 => "open64",
            Self::Openat => "openat",
            Self::Read => "read",
            Self::Readv => "readv",
            Self::Pread => "pread",
            Self::Lseek => "lseek",
            Self::Lseek64 => "lseek64",
            Self::Close => "close",
            Self::Write => "write",
            Self::Writev => "writev",
            Self::Pwrite => "pwrite",
            Self::Ftruncate => "ftruncate",
            Self::Ftruncate64 => "ftruncate64",
            Self::Truncate => "truncate",
            Self::Fsync => "fsync",
            Self::Mkdir => "mkdir",
            Self::Remove => "remove",
            Self::Rename => "rename",
            Self::Unlink => "unlink",
            Self::Link => "link",
            Self::Symlink => "symlink",
            Self::Sendfile => "sendfile",
            Self::Fchmod => "fchmod",
            Self::Fchmodat => "fchmodat",
            Self::Utimensat => "utimensat",
            Self::Xstat => "__xstat",
            Self::Fxstat => "__fxstat",
            Self::Lxstat => "__lxstat",
            Self::Xstat64 => "__xstat64",
            Self::Fxstat64 => "__fxstat64",
            Self::Lxstat64 => "__lxstat64",
            Self::Stat => "stat",
            Self::Fstat => "fstat",
            Self::Lstat => "lstat",
            Self::Fstatat => "fstatat",
            Self::Statx => "statx",
            Self::Statvfs => "statvfs",
            Self::Fstatvfs => "fstatvfs",
            Self::Opendir => "opendir",
            Self::Fdopendir => "fdopendir",
            Self::Readdir => "readdir",
            Self::Readdir64 => "readdir64",
            Self::Closedir => "closedir",
            Self::Realpath => "realpath",
            Self::Access => "access",
            Self::Faccessat => "faccessat",
            Self::Getcwd => "getcwd",
            Self::Chdir => "chdir",
            Self::Readlink => "readlink",
            Self::Dlopen => "dlopen",
            Self::Dlclose => "dlclose",
        }
    }

    /// Resolve a symbol name to its [`Sym`], if `hookfs` knows it.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        ALL.iter().copied().find(|s| s.name() == name)
    }

    /// Total number of known symbols (the size of the [`ORIGINALS`] table).
    pub(crate) const COUNT: usize = 54;
}

/// Every symbol, in discriminant order (also the install candidate list).
#[cfg(target_os = "linux")]
pub(crate) const ALL: [Sym; Sym::COUNT] = [
    Sym::Fopen,
    Sym::Fopen64,
    Sym::Open,
    Sym::Open64,
    Sym::Openat,
    Sym::Read,
    Sym::Readv,
    Sym::Pread,
    Sym::Lseek,
    Sym::Lseek64,
    Sym::Close,
    Sym::Write,
    Sym::Writev,
    Sym::Pwrite,
    Sym::Ftruncate,
    Sym::Ftruncate64,
    Sym::Truncate,
    Sym::Fsync,
    Sym::Mkdir,
    Sym::Remove,
    Sym::Rename,
    Sym::Unlink,
    Sym::Link,
    Sym::Symlink,
    Sym::Sendfile,
    Sym::Fchmod,
    Sym::Fchmodat,
    Sym::Utimensat,
    Sym::Xstat,
    Sym::Fxstat,
    Sym::Lxstat,
    Sym::Xstat64,
    Sym::Fxstat64,
    Sym::Lxstat64,
    Sym::Stat,
    Sym::Fstat,
    Sym::Lstat,
    Sym::Fstatat,
    Sym::Statx,
    Sym::Statvfs,
    Sym::Fstatvfs,
    Sym::Opendir,
    Sym::Fdopendir,
    Sym::Readdir,
    Sym::Readdir64,
    Sym::Closedir,
    Sym::Realpath,
    Sym::Access,
    Sym::Faccessat,
    Sym::Getcwd,
    Sym::Chdir,
    Sym::Readlink,
    Sym::Dlopen,
    Sym::Dlclose,
];

// ---- Darwin (macOS + iOS/iPadOS) symbol table ------------------------------

/// versioned wrappers and **no** `*64` variants (`open`/`fopen`/`lseek`/`stat` are
/// already 64-bit); it uses `stat`/`fstat`/`lstat`/`fstatat` directly.
///
/// # `$INODE64` and iOS
/// The x86-64-macOS legacy `$INODE64` aliases (`stat$INODE64`, `readdir$INODE64`, …)
/// are **not** listed here: the Mach-O parser normalizes a trailing `$…` suffix away
/// (`_stat$INODE64` → `stat`), so a slice importing them matches the plain `Stat` /
/// `Readdir` entries below. On **iOS** (arm64-only, native 64-bit inodes) there is no
/// `$INODE64` suffix at all — the bare `stat`/`readdir` are what a slice imports, and
/// they match directly. So this one set is correct for both without a per-platform
/// split: any name a target does not import is simply skipped (install is optional).
/// `__getdirentries64` is the low-level dir syscall wrapper. `sendfile` (a different,
/// Darwin-specific ABI) is omitted (unused by the BRAW workflow). As on Linux, the
/// post-`fopen` stdio family is deliberately absent — serviced by the `funopen`
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Sym {
    // Open family (routing by path).
    Fopen,
    Open,
    Openat,
    // Descriptor family (routing by registry membership).
    Read,
    Readv,
    Pread,
    Lseek,
    Close,
    // Write / truncate / mutate — fail closed (EROFS) for virtual targets.
    Write,
    Writev,
    Pwrite,
    Ftruncate,
    Truncate,
    Fsync,
    Mkdir,
    Remove,
    Rename,
    Unlink,
    Link,
    Symlink,
    Fchmod,
    Fchmodat,
    Utimensat,
    // Stat family (macOS `struct stat`; `$INODE64` aliases normalize to these).
    Stat,
    Fstat,
    Lstat,
    Fstatat,
    // Volume.
    Statvfs,
    Fstatvfs,
    // Directory family (`readdir$INODE64` normalizes to `Readdir`).
    Opendir,
    Fdopendir,
    Readdir,
    Closedir,
    Getdirentries64,
    // Path.
    Realpath,
    Access,
    Faccessat,
    Getcwd,
    Chdir,
    Readlink,
    // Loader (passthrough + auto_rescan).
    Dlopen,
    Dlclose,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Sym {
    /// The exported symbol name (the leading-`_`/`$INODE64`-stripped base name the
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Fopen => "fopen",
            Self::Open => "open",
            Self::Openat => "openat",
            Self::Read => "read",
            Self::Readv => "readv",
            Self::Pread => "pread",
            Self::Lseek => "lseek",
            Self::Close => "close",
            Self::Write => "write",
            Self::Writev => "writev",
            Self::Pwrite => "pwrite",
            Self::Ftruncate => "ftruncate",
            Self::Truncate => "truncate",
            Self::Fsync => "fsync",
            Self::Mkdir => "mkdir",
            Self::Remove => "remove",
            Self::Rename => "rename",
            Self::Unlink => "unlink",
            Self::Link => "link",
            Self::Symlink => "symlink",
            Self::Fchmod => "fchmod",
            Self::Fchmodat => "fchmodat",
            Self::Utimensat => "utimensat",
            Self::Stat => "stat",
            Self::Fstat => "fstat",
            Self::Lstat => "lstat",
            Self::Fstatat => "fstatat",
            Self::Statvfs => "statvfs",
            Self::Fstatvfs => "fstatvfs",
            Self::Opendir => "opendir",
            Self::Fdopendir => "fdopendir",
            Self::Readdir => "readdir",
            Self::Closedir => "closedir",
            Self::Getdirentries64 => "__getdirentries64",
            Self::Realpath => "realpath",
            Self::Access => "access",
            Self::Faccessat => "faccessat",
            Self::Getcwd => "getcwd",
            Self::Chdir => "chdir",
            Self::Readlink => "readlink",
            Self::Dlopen => "dlopen",
            Self::Dlclose => "dlclose",
        }
    }

    /// Resolve a symbol name to its [`Sym`], if `hookfs` knows it.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        ALL.iter().copied().find(|s| s.name() == name)
    }

    /// Total number of known symbols (the size of the [`ORIGINALS`] table).
    pub(crate) const COUNT: usize = 42;
}

/// Every symbol, in discriminant order (also the install candidate list).
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) const ALL: [Sym; Sym::COUNT] = [
    Sym::Fopen,
    Sym::Open,
    Sym::Openat,
    Sym::Read,
    Sym::Readv,
    Sym::Pread,
    Sym::Lseek,
    Sym::Close,
    Sym::Write,
    Sym::Writev,
    Sym::Pwrite,
    Sym::Ftruncate,
    Sym::Truncate,
    Sym::Fsync,
    Sym::Mkdir,
    Sym::Remove,
    Sym::Rename,
    Sym::Unlink,
    Sym::Link,
    Sym::Symlink,
    Sym::Fchmod,
    Sym::Fchmodat,
    Sym::Utimensat,
    Sym::Stat,
    Sym::Fstat,
    Sym::Lstat,
    Sym::Fstatat,
    Sym::Statvfs,
    Sym::Fstatvfs,
    Sym::Opendir,
    Sym::Fdopendir,
    Sym::Readdir,
    Sym::Closedir,
    Sym::Getdirentries64,
    Sym::Realpath,
    Sym::Access,
    Sym::Faccessat,
    Sym::Getcwd,
    Sym::Chdir,
    Sym::Readlink,
    Sym::Dlopen,
    Sym::Dlclose,
];

/// Saved original function pointers, one atomic cell per [`Sym`], indexed by its
/// discriminant. `0` means "not resolved" (the symbol was not present / hooked).
/// Cells are only ever *set* (to a process-lifetime-valid code address) and never
/// cleared, so a passthrough read is always either a valid pointer or `0`.
static ORIGINALS: [AtomicUsize; Sym::COUNT] = [const { AtomicUsize::new(0) }; Sym::COUNT];

/// Record the canonical original entry point for `name`, if known. Called at
/// install time, before the corresponding slot is patched.
pub(crate) fn set_original(name: &str, ptr: usize) {
    if let Some(cell) = Sym::from_name(name).and_then(|sym| ORIGINALS.get(sym as usize)) {
        cell.store(ptr, Ordering::Release);
    }
}

/// The saved original for `sym`, or `0` if unknown.
pub(crate) fn original(sym: Sym) -> usize {
    ORIGINALS
        .get(sym as usize)
        .map_or(0, |cell| cell.load(Ordering::Acquire))
}

/// The **single active installation** and its routing state. `hookfs` supports at
/// single global engine backs every shim, so overlapping installs would silently
/// hijack each other. Installing publishes exactly one engine here; only the guard
/// that owns it may tear it down. `None` when no installation is active — every
/// shim then passes straight through.
///
/// The engine is cloned out (an `Arc`) for the duration of each call so it stays
/// alive even if the owning guard drops concurrently.
static ACTIVE: RwLock<Option<Active>> = RwLock::new(None);

/// Source of a unique token per successful installation. A guard remembers its
/// token and tears the installation down only if it still owns [`ACTIVE`], so a
/// stale or double drop can never clear a *different* installation's state.
static NEXT_INSTALL_ID: AtomicU64 = AtomicU64::new(1);

/// The active routing engine plus the id of the installation that published it.
struct Active {
    engine: Arc<Engine>,
    id: u64,
}

/// Become the process's single active installation, publishing `engine` as the
/// active routing state and returning a unique installation id. Returns `None`
/// when an installation is already active — the caller must then fail loudly
/// (`Error::AlreadyInstalled`) instead of clobbering the live installation.
pub(crate) fn try_activate(engine: Arc<Engine>) -> Option<u64> {
    let mut slot = ACTIVE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.is_some() {
        return None;
    }
    let id = NEXT_INSTALL_ID.fetch_add(1, Ordering::Relaxed);
    *slot = Some(Active { engine, id });
    Some(id)
}

/// Tear down the active installation, but only if it is still the one identified
/// by `id`. A guard for an already-cleared installation is a no-op, so a stale or
/// double drop can never clear a different installation's state.
pub(crate) fn deactivate(id: u64) {
    let mut slot = ACTIVE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.as_ref().is_some_and(|active| active.id == id) {
        *slot = None;
    }
}

/// A cloned handle to the active engine, or `None` if none is installed.
pub(crate) fn current_engine() -> Option<Arc<Engine>> {
    ACTIVE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(|a| a.engine.clone())
}

/// Serializes the tests that touch the process-global installation/engine state,
/// so they never race on the single-active-installation slot.
#[cfg(test)]
pub(crate) static GLOBAL_STATE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// An optional rescan callback, invoked by the `LoadLibrary*` shims after a
/// successful load when `auto_rescan` is enabled, so late-loaded decoder plugins
static RESCAN: RwLock<Option<Arc<dyn Fn() + Send + Sync>>> = RwLock::new(None);

/// Register the rescan callback (install time, when `auto_rescan` is on).
pub(crate) fn set_rescan(cb: Arc<dyn Fn() + Send + Sync>) {
    *RESCAN
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cb);
}

/// Clear the rescan callback (uninstall).
pub(crate) fn clear_rescan() {
    *RESCAN
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Invoke the rescan callback, if any. Best-effort: a failure inside it must not
/// disturb the `LoadLibrary` the SDK just made.
pub(crate) fn trigger_rescan() {
    let cb = RESCAN
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    if let Some(cb) = cb {
        cb();
    }
}

thread_local! {
    /// Set while this thread is inside a `hookfs` shim body, so a re-entered shim
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// RAII marker that a shim is servicing a call on this thread. [`enter`] returns
/// `None` if the thread is already inside a shim — the caller then passes through.
pub(crate) struct HookScope;

impl HookScope {
    /// Enter the hook body, or `None` if already inside one on this thread.
    pub(crate) fn enter() -> Option<Self> {
        IN_HOOK.with(|flag| {
            if flag.get() {
                None
            } else {
                flag.set(true);
                Some(Self)
            }
        })
    }
}

impl Drop for HookScope {
    fn drop(&mut self) {
        IN_HOOK.with(|flag| flag.set(false));
    }
}

/// Run a shim's Rust work behind a panic firewall. If the closure unwinds, the
/// panic is contained (never crossing the FFI boundary), the OS last-error is set
/// to `on_panic`, and `fallback` is returned.
///
/// # Safety of the ABI contract
/// The returned value is passed straight back to native code, so callers must
/// supply a `fallback` that is a valid failure sentinel for the specific ABI
/// (`INVALID_HANDLE_VALUE`, `FALSE`, `0`, …).
pub(crate) fn guard_abi<R>(fallback: R, on_panic: u32, f: impl FnOnce() -> R) -> R {
    if let Ok(value) = std::panic::catch_unwind(AssertUnwindSafe(f)) {
        value
    } else {
        #[cfg(windows)]
        // SAFETY: `SetLastError` has no preconditions.
        unsafe {
            windows_sys::Win32::Foundation::SetLastError(on_panic);
        }
        // `errno` is written through the platform's thread-local accessor: glibc's
        #[cfg(target_os = "linux")]
        // SAFETY: `__errno_location` returns a valid, thread-local `int*`.
        unsafe {
            *libc::__errno_location() = i32::try_from(on_panic).unwrap_or(libc::EIO);
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        // SAFETY: `__error` returns a valid, thread-local `int*`.
        unsafe {
            *libc::__error() = i32::try_from(on_panic).unwrap_or(libc::EIO);
        }
        fallback
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::namespace::Namespace;
    use crate::providers::MemoryFs;

    fn test_engine() -> Arc<Engine> {
        // Platform-correct reserved root so the test builds on Windows and POSIX.
        let ns = Namespace::from_root(crate::namespace::reserved_root());
        Arc::new(Engine::new(MemoryFs::new(), ns, false))
    }

    #[test]
    fn single_active_installation_is_enforced() {
        let _serial = GLOBAL_STATE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // No installation active: nothing to route.
        assert!(current_engine().is_none());

        // First activation succeeds and publishes the engine.
        let id1 = try_activate(test_engine()).expect("first activation should succeed");
        assert!(current_engine().is_some());

        // A second activation while one is active is refused (no clobber).
        assert!(try_activate(test_engine()).is_none());
        assert!(current_engine().is_some());

        // A stale/foreign token must not tear down the live installation.
        deactivate(id1.wrapping_add(1_000));
        assert!(current_engine().is_some());

        // The owning token tears it down; a second activation then succeeds again.
        deactivate(id1);
        assert!(current_engine().is_none());
        let id2 = try_activate(test_engine()).expect("activation should succeed after teardown");
        assert_ne!(id1, id2, "each installation gets a fresh token");
        deactivate(id2);
        assert!(current_engine().is_none());
    }
}
