//!
//! Each shim is an `extern "system"` function with the *exact* ABI of the
//! KERNEL32 file API it replaces (types cross-checked against `windows-sys`,
//! R13). Every shim:
//!
//! 1. runs its Rust work behind [`guard_abi`] — a panic becomes a native error,
//!    never an unwind across the FFI boundary (R15);
//! 2. honors the thread-local reentrancy guard — a re-entered shim passes through;
//! 3. routes: the open/path family decides virtual-vs-real by **path**, the
//!    handle family by **registry membership** (handles are generic — R3);
//! 4. for a virtual object, serves it from the VFS/registry and sets
//!    `SetLastError` on every failure path; for anything else, calls the saved
//!    **original** pointer so non-virtual paths and handles keep byte-for-byte /
//!    native-error parity.
//!
//! A virtual open returns a **real carrier `HANDLE`** (an opened `NUL` device via
//! the original `CreateFileW`), so a handle that escapes to an unhooked consumer
//! or the kernel fails predictably instead of dereferencing a fabricated value.
//! Fail-closed restrictions: synchronous I/O only (`ERROR_NOT_SUPPORTED` for
//! overlapped/async, R16); writes to virtual paths are denied in the read-only
//! milestone (`ERROR_ACCESS_DENIED`).

#![allow(non_snake_case)] // Shim params mirror the Win32 SAL names for clarity.

use crate::dispatch::{self, HookScope, Sym, current_engine, guard_abi};
use crate::namespace::{decode_ansi, decode_wide, wildcard_match};
use crate::registry::{FindState, OpenFile, OpenStream, VIRTUAL_VOLUME_SERIAL};
use crate::router::Route;
use crate::vfs::{OpenOptions, VfsDirEntry, VfsMetadata};
use core::ffi::c_void;
use std::io::SeekFrom;
use std::path::Path;

use windows_sys::Win32::Foundation::{
    ERROR_ACCESS_DENIED, ERROR_DIR_NOT_EMPTY, ERROR_DIRECTORY, ERROR_FILE_EXISTS,
    ERROR_FILE_NOT_FOUND, ERROR_INTERNAL_ERROR, ERROR_INVALID_NAME, ERROR_INVALID_PARAMETER,
    ERROR_NO_MORE_FILES, ERROR_NOT_SUPPORTED, ERROR_SUCCESS, GENERIC_WRITE, HANDLE,
    INVALID_HANDLE_VALUE, SetLastError,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_NORMAL,
    FILE_ATTRIBUTE_READONLY, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE,
    FILE_TYPE_DISK, FindExInfoStandard, FindExSearchNameMatch, INVALID_FILE_ATTRIBUTES,
    INVALID_SET_FILE_POINTER, OPEN_EXISTING, WIN32_FILE_ATTRIBUTE_DATA, WIN32_FIND_DATAW,
};
use windows_sys::Win32::System::IO::OVERLAPPED;
use windows_sys::Win32::System::SystemInformation::GetWindowsDirectoryW;
use windows_sys::core::{BOOL, PCSTR, PCWSTR, PSTR, PWSTR};

/// `DRIVE_FIXED` — a virtual root reports as a fixed local disk.
const DRIVE_FIXED: u32 = 3;
/// `GENERIC_READ` for opening the carrier handle.
const GENERIC_READ: u32 = 0x8000_0000;
/// The last-error code used when a shim's Rust work panics (R15).
const PANIC_ERROR: u32 = ERROR_INTERNAL_ERROR;

// `dwCreationDisposition` values for `CreateFileW` (`OPEN_EXISTING` == 3 is
// imported from `windows-sys`). Declared locally, like `GENERIC_READ`, to keep
// the shim's Win32 constant surface explicit.
const CREATE_NEW: u32 = 1;
const CREATE_ALWAYS: u32 = 2;
const OPEN_ALWAYS: u32 = 4;
const TRUNCATE_EXISTING: u32 = 5;

/// The `dwDesiredAccess` bits that request write access: `GENERIC_WRITE`,
/// `GENERIC_ALL`, and the specific `FILE_WRITE_DATA`/`FILE_APPEND_DATA` rights.
const WRITE_ACCESS_MASK: u32 = GENERIC_WRITE | 0x1000_0000 | 0x0002 | 0x0004;
/// `FILE_APPEND_DATA` alone (without `FILE_WRITE_DATA`) selects append semantics.
const FILE_APPEND_DATA: u32 = 0x0004;
const FILE_WRITE_DATA: u32 = 0x0002;

// ---- Original function-pointer types (exact ABI) ---------------------------

type PfnCreateFileW =
    unsafe extern "system" fn(PCWSTR, u32, u32, *const c_void, u32, u32, HANDLE) -> HANDLE;
type PfnCreateFileA =
    unsafe extern "system" fn(PCSTR, u32, u32, *const c_void, u32, u32, HANDLE) -> HANDLE;
type PfnCreateFile2 = unsafe extern "system" fn(PCWSTR, u32, u32, u32, *const c_void) -> HANDLE;
type PfnReadFile =
    unsafe extern "system" fn(HANDLE, *mut u8, u32, *mut u32, *mut OVERLAPPED) -> BOOL;
type PfnReadFileEx =
    unsafe extern "system" fn(HANDLE, *mut u8, u32, *mut OVERLAPPED, *const c_void) -> BOOL;
type PfnWriteFile =
    unsafe extern "system" fn(HANDLE, *const u8, u32, *mut u32, *mut OVERLAPPED) -> BOOL;
type PfnWriteFileEx =
    unsafe extern "system" fn(HANDLE, *const u8, u32, *mut OVERLAPPED, *const c_void) -> BOOL;
type PfnSetFilePointer = unsafe extern "system" fn(HANDLE, i32, *mut i32, u32) -> u32;
type PfnSetFilePointerEx = unsafe extern "system" fn(HANDLE, i64, *mut i64, u32) -> BOOL;
type PfnGetFileSize = unsafe extern "system" fn(HANDLE, *mut u32) -> u32;
type PfnGetFileSizeEx = unsafe extern "system" fn(HANDLE, *mut i64) -> BOOL;
type PfnGetFileType = unsafe extern "system" fn(HANDLE) -> u32;
type PfnGetFileInformationByHandle =
    unsafe extern "system" fn(HANDLE, *mut BY_HANDLE_FILE_INFORMATION) -> BOOL;
type PfnGetFileInformationByHandleEx =
    unsafe extern "system" fn(HANDLE, i32, *mut c_void, u32) -> BOOL;
type PfnFlushFileBuffers = unsafe extern "system" fn(HANDLE) -> BOOL;
type PfnSetEndOfFile = unsafe extern "system" fn(HANDLE) -> BOOL;
type PfnCloseHandle = unsafe extern "system" fn(HANDLE) -> BOOL;
type PfnFindFirstFileExW =
    unsafe extern "system" fn(PCWSTR, i32, *mut c_void, i32, *const c_void, u32) -> HANDLE;
type PfnFindFirstFileW = unsafe extern "system" fn(PCWSTR, *mut WIN32_FIND_DATAW) -> HANDLE;
type PfnFindNextFileW = unsafe extern "system" fn(HANDLE, *mut WIN32_FIND_DATAW) -> BOOL;
type PfnFindClose = unsafe extern "system" fn(HANDLE) -> BOOL;
type PfnGetFileAttributesW = unsafe extern "system" fn(PCWSTR) -> u32;
type PfnGetFileAttributesExW = unsafe extern "system" fn(PCWSTR, i32, *mut c_void) -> BOOL;
type PfnGetFullPathNameW = unsafe extern "system" fn(PCWSTR, u32, PWSTR, *mut PWSTR) -> u32;
type PfnGetFullPathNameA = unsafe extern "system" fn(PCSTR, u32, PSTR, *mut PSTR) -> u32;
type PfnGetCurrentDirectoryW = unsafe extern "system" fn(u32, PWSTR) -> u32;
type PfnDeleteFileW = unsafe extern "system" fn(PCWSTR) -> BOOL;
type PfnMoveFileExW = unsafe extern "system" fn(PCWSTR, PCWSTR, u32) -> BOOL;
type PfnGetDiskFreeSpaceA =
    unsafe extern "system" fn(PCSTR, *mut u32, *mut u32, *mut u32, *mut u32) -> BOOL;
type PfnGetDriveTypeW = unsafe extern "system" fn(PCWSTR) -> u32;
type PfnCreateDirectoryW = unsafe extern "system" fn(PCWSTR, *const c_void) -> BOOL;
type PfnLoadLibraryExW = unsafe extern "system" fn(PCWSTR, HANDLE, u32) -> HANDLE;
type PfnLoadLibraryA = unsafe extern "system" fn(PCSTR) -> HANDLE;

/// The first three fields of `CREATEFILE2_EXTENDED_PARAMETERS`, enough to read the
/// requested file flags without pulling the `Win32_Security` feature in.
#[repr(C)]
#[allow(clippy::struct_field_names)] // mirrors the Win32 `dw*` field names.
struct CreateFile2Head {
    dw_size: u32,
    dw_file_attributes: u32,
    dw_file_flags: u32,
}

/// `FILE_STANDARD_INFO` (class 1) — declared locally to avoid a feature pull.
#[repr(C)]
struct FileStandardInfoData {
    allocation_size: i64,
    end_of_file: i64,
    number_of_links: u32,
    delete_pending: u8,
    directory: u8,
}

/// `FILE_BASIC_INFO` (class 0).
#[repr(C)]
struct FileBasicInfoData {
    creation_time: i64,
    last_access_time: i64,
    last_write_time: i64,
    change_time: i64,
    file_attributes: u32,
}

// ---- Small helpers ---------------------------------------------------------

/// Transmute the saved original of `sym` into its function-pointer type.
#[inline]
fn orig_fn<T: Copy>(sym: Sym) -> T {
    let ptr = dispatch::original(sym);
    debug_assert_ne!(
        ptr, 0,
        "original for {sym:?} was not resolved before the hook ran"
    );
    // SAFETY: `T` is an `extern "system"` fn-pointer type (pointer-sized), and
    // `ptr` is the canonical KERNEL32 export resolved at install time.
    unsafe { core::mem::transmute_copy::<usize, T>(&ptr) }
}

/// Set the OS thread-local last-error.
#[inline]
fn set_err(code: u32) {
    // SAFETY: `SetLastError` has no preconditions.
    unsafe { SetLastError(code) }
}

/// Map an `io::Error` to the closest Win32 error code.
fn io_to_win(err: &std::io::Error) -> u32 {
    // A native Win32 code carried by the error wins (e.g. `ERROR_DISK_FULL` from a
    // full synthetic volume, produced via `vfs::no_space`).
    if let Some(code) = err.raw_os_error()
        && let Ok(code) = u32::try_from(code)
        && code != 0
    {
        return code;
    }
    match err.kind() {
        std::io::ErrorKind::NotFound => ERROR_FILE_NOT_FOUND,
        // Win32 has no dedicated "is a directory" code; opening a directory as a
        // file yields ACCESS_DENIED, so it shares the PermissionDenied mapping.
        std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::IsADirectory => {
            ERROR_ACCESS_DENIED
        }
        std::io::ErrorKind::AlreadyExists => ERROR_FILE_EXISTS,
        std::io::ErrorKind::InvalidInput => ERROR_INVALID_PARAMETER,
        std::io::ErrorKind::DirectoryNotEmpty => ERROR_DIR_NOT_EMPTY,
        std::io::ErrorKind::NotADirectory => ERROR_DIRECTORY,
        _ => ERROR_INTERNAL_ERROR,
    }
}

/// Split a `u64` into its high and low 32-bit halves without a truncating cast.
fn split_hi_lo(value: u64) -> (u32, u32) {
    let hi = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    (hi, lo)
}

/// A stable 64-bit file identity for a virtual path, used to fill the
/// `nFileIndexHigh`/`nFileIndexLow` of `BY_HANDLE_FILE_INFORMATION`. Derived from
/// virtual path report the same identity** — a consumer that dedups files by
/// (`dwVolumeSerialNumber`, `nFileIndex`) sees one virtual file as one file —
/// while distinct paths stay distinct with high probability. The hasher uses fixed
/// keys, so the identity is deterministic (unlike a per-open counter or a
/// randomly-seeded hash). `0` is remapped, as some consumers treat it as "no id".
fn stable_file_id(path: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    crate::namespace::normalize_key(path).hash(&mut hasher);
    match hasher.finish() {
        0 => 1,
        id => id,
    }
}

/// Open a real, harmless carrier `HANDLE` (the `NUL` device) through the original
/// `CreateFileW`. Returns `None` if the device could not be opened.
fn carrier_open() -> Option<HANDLE> {
    let nul: [u16; 4] = [u16::from(b'N'), u16::from(b'U'), u16::from(b'L'), 0];
    let create: PfnCreateFileW = orig_fn(Sym::CreateFileW);
    // SAFETY: valid NUL-terminated wide name; all other args are plain scalars.
    let handle = unsafe {
        create(
            nul.as_ptr(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            core::ptr::null(),
            OPEN_EXISTING,
            0,
            core::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE || handle.is_null() {
        None
    } else {
        Some(handle)
    }
}

/// Open a real carrier *find* `HANDLE` by enumerating the Windows directory
/// through the original `FindFirstFileExW`. The real result is discarded; only the
/// handle is kept, so a stray `FindNextFileW`/`FindClose` on it stays valid.
fn carrier_find() -> Option<HANDLE> {
    let mut buf = [0u16; 260];
    // SAFETY: `buf` is a valid writable buffer of 260 wide chars.
    let len = unsafe { GetWindowsDirectoryW(buf.as_mut_ptr(), 260) } as usize;
    let dir = buf.get(..len)?;
    let mut pattern: Vec<u16> = dir.to_vec();
    pattern.push(u16::from(b'\\'));
    pattern.push(u16::from(b'*'));
    pattern.push(0);

    // SAFETY: plain-old-data; fully overwritten (or left zeroed) by the callee.
    let mut data: WIN32_FIND_DATAW = unsafe { core::mem::zeroed() };
    let find: PfnFindFirstFileExW = orig_fn(Sym::FindFirstFileExW);
    // SAFETY: valid NUL-terminated pattern; `data` is a valid out buffer.
    let handle = unsafe {
        find(
            pattern.as_ptr(),
            FindExInfoStandard,
            core::ptr::from_mut(&mut data).cast(),
            FindExSearchNameMatch,
            core::ptr::null(),
            0,
        )
    };
    if handle == INVALID_HANDLE_VALUE || handle.is_null() {
        None
    } else {
        Some(handle)
    }
}

/// The synthetic Win32 file attributes for a VFS entry.
fn attrs_for(meta: &VfsMetadata) -> u32 {
    if meta.is_dir {
        FILE_ATTRIBUTE_DIRECTORY
    } else if meta.readonly {
        FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_READONLY
    } else {
        FILE_ATTRIBUTE_NORMAL
    }
}

/// Fill a `WIN32_FIND_DATAW` from a directory entry.
///
/// # Safety
/// `out` must be a valid, writable `WIN32_FIND_DATAW`.
unsafe fn fill_find_data(out: *mut WIN32_FIND_DATAW, entry: &VfsDirEntry) {
    if out.is_null() {
        return;
    }
    // SAFETY: POD struct; zero is a valid bit pattern for every field.
    let mut data: WIN32_FIND_DATAW = unsafe { core::mem::zeroed() };
    data.dwFileAttributes = attrs_for(&entry.metadata);
    let (hi, lo) = split_hi_lo(entry.metadata.len);
    data.nFileSizeHigh = hi;
    data.nFileSizeLow = lo;
    let name: Vec<u16> = entry
        .name
        .to_string_lossy()
        .encode_utf16()
        .take(259)
        .collect();
    for (dst, src) in data.cFileName.iter_mut().zip(name.iter()) {
        *dst = *src;
    }
    // SAFETY: caller guarantees `out` is writable.
    unsafe { *out = data };
}

/// Route a decoded path via the active engine. Returns `None` when there is no
/// engine or the shim is re-entered (the caller then passes through).
fn route(decoded: &str) -> Option<(std::sync::Arc<crate::router::Engine>, Route)> {
    let engine = current_engine()?;
    let route = engine.classify(decoded);
    Some((engine, route))
}

// ---- Open family -----------------------------------------------------------

/// Build [`OpenOptions`] from a `CreateFileW` desired access + creation disposition
/// whether a missing/existing file is an error.
fn open_options(desired_access: u32, disposition: u32) -> OpenOptions {
    let write = desired_access & WRITE_ACCESS_MASK != 0;
    // Append when `FILE_APPEND_DATA` is requested without `FILE_WRITE_DATA`.
    let append = desired_access & FILE_APPEND_DATA != 0 && desired_access & FILE_WRITE_DATA == 0;
    let (create, create_new, truncate) = match disposition {
        CREATE_NEW => (false, true, false),   // create; fail if it exists
        CREATE_ALWAYS => (true, false, true), // create; truncate if it exists
        OPEN_ALWAYS => (true, false, false),  // open; create if missing
        TRUNCATE_EXISTING => (false, false, true), // open existing; truncate
        _ => (false, false, false),           // OPEN_EXISTING (and unknown)
    };
    OpenOptions {
        read: true,
        write,
        create,
        create_new,
        truncate,
        append,
    }
}

/// Shared virtual-open logic for `CreateFileW`/`A`/`2`. On the **write** path (a
/// write access bit set) with writes permitted, it drives the VFS `open_write`
/// (create/truncate/append per the disposition); otherwise it serves a read view.
/// A write request with writes disabled fails closed (`ERROR_ACCESS_DENIED`).
fn open_virtual(
    engine: &crate::router::Engine,
    path: &Path,
    desired_access: u32,
    disposition: u32,
    flags: u32,
) -> HANDLE {
    let opts = open_options(desired_access, disposition);
    if opts.write && !engine.allow_writes() {
        // Writes disabled: fail closed rather than silently serving read-only.
        set_err(ERROR_ACCESS_DENIED);
        return INVALID_HANDLE_VALUE;
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
            // Virtual path the VFS does not have (and will not create): fail closed.
            set_err(ERROR_FILE_NOT_FOUND);
            INVALID_HANDLE_VALUE
        }
        Some(Err(err)) => {
            set_err(io_to_win(&err));
            INVALID_HANDLE_VALUE
        }
        Some(Ok(stream)) => {
            let Some(carrier) = carrier_open() else {
                set_err(ERROR_INTERNAL_ERROR);
                return INVALID_HANDLE_VALUE;
            };
            engine.registry().insert_file(
                carrier as usize,
                OpenFile {
                    stream,
                    path: path.to_owned(),
                    writable: opts.write,
                    overlapped: flags & FILE_FLAG_OVERLAPPED != 0,
                },
            );
            set_err(ERROR_SUCCESS);
            carrier
        }
    }
}

/// `CreateFileW` shim.
pub(crate) extern "system" fn shim_CreateFileW(
    lpFileName: PCWSTR,
    dwDesiredAccess: u32,
    dwShareMode: u32,
    lpSecurityAttributes: *const c_void,
    dwCreationDisposition: u32,
    dwFlagsAndAttributes: u32,
    hTemplateFile: HANDLE,
) -> HANDLE {
    guard_abi(INVALID_HANDLE_VALUE, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnCreateFileW = orig_fn(Sym::CreateFileW);
            // SAFETY: forwarding the caller's exact arguments to the real API.
            unsafe {
                orig(
                    lpFileName,
                    dwDesiredAccess,
                    dwShareMode,
                    lpSecurityAttributes,
                    dwCreationDisposition,
                    dwFlagsAndAttributes,
                    hTemplateFile,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: `lpFileName` is the caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_FILE_NOT_FOUND);
                INVALID_HANDLE_VALUE
            }
            Some((engine, Route::Virtual(path))) => open_virtual(
                &engine,
                &path,
                dwDesiredAccess,
                dwCreationDisposition,
                dwFlagsAndAttributes,
            ),
        }
    })
}

/// `CreateFileA` shim.
pub(crate) extern "system" fn shim_CreateFileA(
    lpFileName: PCSTR,
    dwDesiredAccess: u32,
    dwShareMode: u32,
    lpSecurityAttributes: *const c_void,
    dwCreationDisposition: u32,
    dwFlagsAndAttributes: u32,
    hTemplateFile: HANDLE,
) -> HANDLE {
    guard_abi(INVALID_HANDLE_VALUE, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnCreateFileA = orig_fn(Sym::CreateFileA);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    lpFileName,
                    dwDesiredAccess,
                    dwShareMode,
                    lpSecurityAttributes,
                    dwCreationDisposition,
                    dwFlagsAndAttributes,
                    hTemplateFile,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated byte path (or null).
        let Some(decoded) = (unsafe { decode_ansi(lpFileName) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_FILE_NOT_FOUND);
                INVALID_HANDLE_VALUE
            }
            Some((engine, Route::Virtual(path))) => open_virtual(
                &engine,
                &path,
                dwDesiredAccess,
                dwCreationDisposition,
                dwFlagsAndAttributes,
            ),
        }
    })
}

/// `CreateFile2` shim.
pub(crate) extern "system" fn shim_CreateFile2(
    lpFileName: PCWSTR,
    dwDesiredAccess: u32,
    dwShareMode: u32,
    dwCreationDisposition: u32,
    pCreateExParams: *const c_void,
) -> HANDLE {
    guard_abi(INVALID_HANDLE_VALUE, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnCreateFile2 = orig_fn(Sym::CreateFile2);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    lpFileName,
                    dwDesiredAccess,
                    dwShareMode,
                    dwCreationDisposition,
                    pCreateExParams,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        let flags = if pCreateExParams.is_null() {
            0
        } else {
            // SAFETY: the caller passes a valid pointer to a struct at least as
            // large as its first three fields.
            unsafe { (*pCreateExParams.cast::<CreateFile2Head>()).dw_file_flags }
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_FILE_NOT_FOUND);
                INVALID_HANDLE_VALUE
            }
            Some((engine, Route::Virtual(path))) => open_virtual(
                &engine,
                &path,
                dwDesiredAccess,
                dwCreationDisposition,
                flags,
            ),
        }
    })
}

// ---- Read / write ----------------------------------------------------------

/// `ReadFile` shim.
pub(crate) extern "system" fn shim_ReadFile(
    hFile: HANDLE,
    lpBuffer: *mut u8,
    nNumberOfBytesToRead: u32,
    lpNumberOfBytesRead: *mut u32,
    lpOverlapped: *mut OVERLAPPED,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnReadFile = orig_fn(Sym::ReadFile);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    hFile,
                    lpBuffer,
                    nNumberOfBytesToRead,
                    lpNumberOfBytesRead,
                    lpOverlapped,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };

        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if open.overlapped {
            // Async handle: not supported yet (R16).
            set_err(ERROR_NOT_SUPPORTED);
            return 0;
        }
        // A non-null OVERLAPPED on a synchronous handle is a *positioned* read:
        if !lpOverlapped.is_null() {
            // SAFETY: caller-provided valid OVERLAPPED.
            let ov = unsafe { &*lpOverlapped };
            // SAFETY: reading the Offset/OffsetHigh arm of the union.
            let (lo, hi) = unsafe {
                (
                    ov.Anonymous.Anonymous.Offset,
                    ov.Anonymous.Anonymous.OffsetHigh,
                )
            };
            let offset = (u64::from(hi) << 32) | u64::from(lo);
            if let Err(err) = open.stream.seek(SeekFrom::Start(offset)) {
                set_err(io_to_win(&err));
                return 0;
            }
        }
        // A zero-length read is a successful no-op. Handle it *before* forming a
        // slice: `slice::from_raw_parts_mut` requires a non-null, aligned pointer
        // even for length 0, so `from_raw_parts_mut(null, 0)` would be UB. The real
        // Win32 `ReadFile` does not touch the buffer for a 0-byte read — it returns
        // TRUE and reports 0 bytes read (verified against KERNEL32) — regardless of
        // whether the buffer pointer is null.
        if nNumberOfBytesToRead == 0 {
            if !lpNumberOfBytesRead.is_null() {
                // SAFETY: caller-provided valid out pointer.
                unsafe { *lpNumberOfBytesRead = 0 };
            }
            set_err(ERROR_SUCCESS);
            return 1;
        }
        // A null destination for a non-zero read is a caller error.
        if lpBuffer.is_null() {
            set_err(ERROR_INVALID_PARAMETER);
            return 0;
        }
        // SAFETY: `lpBuffer` is non-null (checked) and the caller guarantees it
        // addresses `nNumberOfBytesToRead` writable bytes.
        let buf =
            unsafe { core::slice::from_raw_parts_mut(lpBuffer, nNumberOfBytesToRead as usize) };
        match open.stream.read(buf) {
            Ok(n) => {
                if !lpNumberOfBytesRead.is_null() {
                    let count = u32::try_from(n).unwrap_or(nNumberOfBytesToRead);
                    // SAFETY: caller-provided valid out pointer.
                    unsafe { *lpNumberOfBytesRead = count };
                }
                set_err(ERROR_SUCCESS);
                1
            }
            Err(err) => {
                set_err(io_to_win(&err));
                0
            }
        }
    })
}

/// `ReadFileEx` shim — overlapped/async only; unsupported for virtual handles.
pub(crate) extern "system" fn shim_ReadFileEx(
    hFile: HANDLE,
    lpBuffer: *mut u8,
    nNumberOfBytesToRead: u32,
    lpOverlapped: *mut OVERLAPPED,
    lpCompletionRoutine: *const c_void,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnReadFileEx = orig_fn(Sym::ReadFileEx);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    hFile,
                    lpBuffer,
                    nNumberOfBytesToRead,
                    lpOverlapped,
                    lpCompletionRoutine,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().get_file(hFile as usize).is_some() {
            set_err(ERROR_NOT_SUPPORTED); // async I/O on a virtual handle (R16)
            0
        } else {
            pass()
        }
    })
}

/// `WriteFile` shim — writes to a writable virtual handle (honoring a positioned
/// `OVERLAPPED.Offset` on a synchronous handle); `ERROR_ACCESS_DENIED` for a
/// read-only virtual handle; passthrough for a non-virtual handle.
pub(crate) extern "system" fn shim_WriteFile(
    hFile: HANDLE,
    lpBuffer: *const u8,
    nNumberOfBytesToWrite: u32,
    lpNumberOfBytesWritten: *mut u32,
    lpOverlapped: *mut OVERLAPPED,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnWriteFile = orig_fn(Sym::WriteFile);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    hFile,
                    lpBuffer,
                    nNumberOfBytesToWrite,
                    lpNumberOfBytesWritten,
                    lpOverlapped,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };

        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !open.writable {
            set_err(ERROR_ACCESS_DENIED);
            return 0;
        }
        if open.overlapped {
            set_err(ERROR_NOT_SUPPORTED); // async write on a virtual handle (R16)
            return 0;
        }
        // A non-null OVERLAPPED on a synchronous handle is a *positioned* write.
        if !lpOverlapped.is_null() {
            // SAFETY: caller-provided valid OVERLAPPED.
            let ov = unsafe { &*lpOverlapped };
            // SAFETY: reading the Offset/OffsetHigh arm of the union.
            let (lo, hi) = unsafe {
                (
                    ov.Anonymous.Anonymous.Offset,
                    ov.Anonymous.Anonymous.OffsetHigh,
                )
            };
            let offset = (u64::from(hi) << 32) | u64::from(lo);
            if let Err(err) = open.stream.seek(SeekFrom::Start(offset)) {
                set_err(io_to_win(&err));
                return 0;
            }
        }
        // A zero-length write is a successful no-op that does not touch the buffer.
        if nNumberOfBytesToWrite == 0 {
            if !lpNumberOfBytesWritten.is_null() {
                // SAFETY: caller-provided valid out pointer.
                unsafe { *lpNumberOfBytesWritten = 0 };
            }
            set_err(ERROR_SUCCESS);
            return 1;
        }
        if lpBuffer.is_null() {
            set_err(ERROR_INVALID_PARAMETER);
            return 0;
        }
        // SAFETY: `lpBuffer` is non-null (checked) and the caller guarantees it
        // addresses `nNumberOfBytesToWrite` readable bytes.
        let buf = unsafe { core::slice::from_raw_parts(lpBuffer, nNumberOfBytesToWrite as usize) };
        match open.stream.write(buf) {
            Ok(n) => {
                if !lpNumberOfBytesWritten.is_null() {
                    let count = u32::try_from(n).unwrap_or(nNumberOfBytesToWrite);
                    // SAFETY: caller-provided valid out pointer.
                    unsafe { *lpNumberOfBytesWritten = count };
                }
                set_err(ERROR_SUCCESS);
                1
            }
            Err(err) => {
                set_err(io_to_win(&err));
                0
            }
        }
    })
}

/// `WriteFileEx` shim — async + read-only: unsupported/denied for virtual handles.
pub(crate) extern "system" fn shim_WriteFileEx(
    hFile: HANDLE,
    lpBuffer: *const u8,
    nNumberOfBytesToWrite: u32,
    lpOverlapped: *mut OVERLAPPED,
    lpCompletionRoutine: *const c_void,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnWriteFileEx = orig_fn(Sym::WriteFileEx);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    hFile,
                    lpBuffer,
                    nNumberOfBytesToWrite,
                    lpOverlapped,
                    lpCompletionRoutine,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().get_file(hFile as usize).is_some() {
            set_err(ERROR_ACCESS_DENIED);
            0
        } else {
            pass()
        }
    })
}

// ---- Seek / size / type ----------------------------------------------------

/// `SetFilePointerEx` shim.
pub(crate) extern "system" fn shim_SetFilePointerEx(
    hFile: HANDLE,
    liDistanceToMove: i64,
    lpNewFilePointer: *mut i64,
    dwMoveMethod: u32,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnSetFilePointerEx = orig_fn(Sym::SetFilePointerEx);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile, liDistanceToMove, lpNewFilePointer, dwMoveMethod) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };

        let Some(from) = seek_from(dwMoveMethod, liDistanceToMove) else {
            set_err(ERROR_INVALID_PARAMETER);
            return 0;
        };
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match open.stream.seek(from) {
            Ok(pos) => {
                if !lpNewFilePointer.is_null() {
                    // SAFETY: caller-provided valid out pointer.
                    unsafe { *lpNewFilePointer = i64::try_from(pos).unwrap_or(i64::MAX) };
                }
                set_err(ERROR_SUCCESS);
                1
            }
            Err(err) => {
                set_err(io_to_win(&err));
                0
            }
        }
    })
}

/// `SetFilePointer` (32-bit) shim.
pub(crate) extern "system" fn shim_SetFilePointer(
    hFile: HANDLE,
    lDistanceToMove: i32,
    lpDistanceToMoveHigh: *mut i32,
    dwMoveMethod: u32,
) -> u32 {
    guard_abi(INVALID_SET_FILE_POINTER, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnSetFilePointer = orig_fn(Sym::SetFilePointer);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile, lDistanceToMove, lpDistanceToMoveHigh, dwMoveMethod) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };

        // Combine the 32-bit low and optional high halves into a signed 64-bit move.
        let distance: i64 = if lpDistanceToMoveHigh.is_null() {
            i64::from(lDistanceToMove)
        } else {
            // SAFETY: caller-provided valid pointer to the high dword.
            let high = unsafe { *lpDistanceToMoveHigh };
            (i64::from(high) << 32) | i64::from(u32::from_ne_bytes(lDistanceToMove.to_ne_bytes()))
        };
        let Some(from) = seek_from(dwMoveMethod, distance) else {
            set_err(ERROR_INVALID_PARAMETER);
            return INVALID_SET_FILE_POINTER;
        };
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match open.stream.seek(from) {
            Ok(pos) => {
                let (hi, lo) = split_hi_lo(pos);
                if !lpDistanceToMoveHigh.is_null() {
                    // SAFETY: caller-provided valid out pointer.
                    unsafe { *lpDistanceToMoveHigh = i32::from_ne_bytes(hi.to_ne_bytes()) };
                }
                set_err(ERROR_SUCCESS);
                lo
            }
            Err(err) => {
                set_err(io_to_win(&err));
                INVALID_SET_FILE_POINTER
            }
        }
    })
}

/// Translate a Win32 move method + distance into a [`SeekFrom`].
fn seek_from(method: u32, distance: i64) -> Option<SeekFrom> {
    match method {
        0 => u64::try_from(distance).ok().map(SeekFrom::Start), // FILE_BEGIN
        1 => Some(SeekFrom::Current(distance)),                 // FILE_CURRENT
        2 => Some(SeekFrom::End(distance)),                     // FILE_END
        _ => None,
    }
}

/// `GetFileSizeEx` shim.
pub(crate) extern "system" fn shim_GetFileSizeEx(hFile: HANDLE, lpFileSize: *mut i64) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFileSizeEx = orig_fn(Sym::GetFileSizeEx);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile, lpFileSize) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match open.stream.size() {
            Ok(size) => {
                if !lpFileSize.is_null() {
                    // SAFETY: caller-provided valid out pointer.
                    unsafe { *lpFileSize = i64::try_from(size).unwrap_or(i64::MAX) };
                }
                set_err(ERROR_SUCCESS);
                1
            }
            Err(err) => {
                set_err(io_to_win(&err));
                0
            }
        }
    })
}

/// `GetFileSize` (32-bit) shim.
pub(crate) extern "system" fn shim_GetFileSize(hFile: HANDLE, lpFileSizeHigh: *mut u32) -> u32 {
    guard_abi(INVALID_FILE_ATTRIBUTES, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFileSize = orig_fn(Sym::GetFileSize);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile, lpFileSizeHigh) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match open.stream.size() {
            Ok(size) => {
                let (hi, lo) = split_hi_lo(size);
                if !lpFileSizeHigh.is_null() {
                    // SAFETY: caller-provided valid out pointer.
                    unsafe { *lpFileSizeHigh = hi };
                }
                set_err(ERROR_SUCCESS);
                lo
            }
            Err(err) => {
                set_err(io_to_win(&err));
                INVALID_FILE_ATTRIBUTES // INVALID_FILE_SIZE == 0xFFFFFFFF
            }
        }
    })
}

/// `GetFileType` shim.
pub(crate) extern "system" fn shim_GetFileType(hFile: HANDLE) -> u32 {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFileType = orig_fn(Sym::GetFileType);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().get_file(hFile as usize).is_some() {
            set_err(ERROR_SUCCESS);
            FILE_TYPE_DISK
        } else {
            pass()
        }
    })
}

// ---- Metadata by handle ----------------------------------------------------

/// `GetFileInformationByHandle` shim.
pub(crate) extern "system" fn shim_GetFileInformationByHandle(
    hFile: HANDLE,
    lpFileInformation: *mut BY_HANDLE_FILE_INFORMATION,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFileInformationByHandle = orig_fn(Sym::GetFileInformationByHandle);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile, lpFileInformation) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };
        if lpFileInformation.is_null() {
            set_err(ERROR_INVALID_PARAMETER);
            return 0;
        }
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let size = open.stream.size().unwrap_or(0);
        let (size_hi, size_lo) = split_hi_lo(size);
        // Stable, path-derived file identity: two opens of the same virtual path
        // report the same `nFileIndex` (and volume serial), so a consumer that
        // dedups by file id treats one virtual file as one file.
        let (idx_hi, idx_lo) = split_hi_lo(stable_file_id(&open.path));
        let attrs = if open.writable {
            FILE_ATTRIBUTE_NORMAL
        } else {
            FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_READONLY
        };
        // SAFETY: POD out struct; zeroed then fully initialized.
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { core::mem::zeroed() };
        info.dwFileAttributes = attrs;
        info.dwVolumeSerialNumber = VIRTUAL_VOLUME_SERIAL;
        info.nFileSizeHigh = size_hi;
        info.nFileSizeLow = size_lo;
        info.nNumberOfLinks = 1;
        info.nFileIndexHigh = idx_hi;
        info.nFileIndexLow = idx_lo;
        // SAFETY: caller-provided valid out pointer.
        unsafe { *lpFileInformation = info };
        set_err(ERROR_SUCCESS);
        1
    })
}

/// `GetFileInformationByHandleEx` shim. Serves `FileBasicInfo` (0) and
/// `FileStandardInfo` (1) for virtual handles; other classes fail closed.
pub(crate) extern "system" fn shim_GetFileInformationByHandleEx(
    hFile: HANDLE,
    FileInformationClass: i32,
    lpFileInformation: *mut c_void,
    dwBufferSize: u32,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFileInformationByHandleEx = orig_fn(Sym::GetFileInformationByHandleEx);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile, FileInformationClass, lpFileInformation, dwBufferSize) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let size = open.stream.size().unwrap_or(0);
        let need = |bytes: usize| -> bool {
            !lpFileInformation.is_null() && (dwBufferSize as usize) >= bytes
        };
        match FileInformationClass {
            0 => {
                // FileBasicInfo
                if !need(core::mem::size_of::<FileBasicInfoData>()) {
                    set_err(ERROR_INVALID_PARAMETER);
                    return 0;
                }
                let data = FileBasicInfoData {
                    creation_time: 0,
                    last_access_time: 0,
                    last_write_time: 0,
                    change_time: 0,
                    file_attributes: if open.writable {
                        FILE_ATTRIBUTE_NORMAL
                    } else {
                        FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_READONLY
                    },
                };
                // SAFETY: buffer is non-null and large enough (checked above).
                unsafe { *lpFileInformation.cast::<FileBasicInfoData>() = data };
                set_err(ERROR_SUCCESS);
                1
            }
            1 => {
                // FileStandardInfo
                if !need(core::mem::size_of::<FileStandardInfoData>()) {
                    set_err(ERROR_INVALID_PARAMETER);
                    return 0;
                }
                let signed = i64::try_from(size).unwrap_or(i64::MAX);
                let data = FileStandardInfoData {
                    allocation_size: signed,
                    end_of_file: signed,
                    number_of_links: 1,
                    delete_pending: 0,
                    directory: 0,
                };
                // SAFETY: buffer is non-null and large enough (checked above).
                unsafe { *lpFileInformation.cast::<FileStandardInfoData>() = data };
                set_err(ERROR_SUCCESS);
                1
            }
            _ => {
                // Unsupported info class on a virtual handle: fail closed.
                set_err(ERROR_NOT_SUPPORTED);
                0
            }
        }
    })
}

// ---- Flush / truncate / close ----------------------------------------------

/// `FlushFileBuffers` shim — flush a writable virtual handle (a no-op success for a
/// read-only one); passthrough for a non-virtual handle.
pub(crate) extern "system" fn shim_FlushFileBuffers(hFile: HANDLE) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnFlushFileBuffers = orig_fn(Sym::FlushFileBuffers);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match open.stream.flush() {
            Ok(()) => {
                set_err(ERROR_SUCCESS);
                1
            }
            Err(err) => {
                set_err(io_to_win(&err));
                0
            }
        }
    })
}

/// `SetEndOfFile` shim — truncate/extend a writable virtual handle to its current
/// file pointer; `ERROR_ACCESS_DENIED` for a read-only virtual handle; passthrough
/// for a non-virtual handle.
pub(crate) extern "system" fn shim_SetEndOfFile(hFile: HANDLE) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnSetEndOfFile = orig_fn(Sym::SetEndOfFile);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFile) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(file) = engine.registry().get_file(hFile as usize) else {
            return pass();
        };
        let mut open = file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !open.writable {
            set_err(ERROR_ACCESS_DENIED);
            return 0;
        }
        // SetEndOfFile moves the end of file to the current file pointer.
        let pos = match open.stream.stream_position() {
            Ok(pos) => pos,
            Err(err) => {
                set_err(io_to_win(&err));
                return 0;
            }
        };
        match open.stream.set_len(pos) {
            Ok(()) => {
                set_err(ERROR_SUCCESS);
                1
            }
            Err(err) => {
                set_err(io_to_win(&err));
                0
            }
        }
    })
}

/// `CloseHandle` shim — drops the registry entry once, then closes the carrier.
pub(crate) extern "system" fn shim_CloseHandle(hObject: HANDLE) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnCloseHandle = orig_fn(Sym::CloseHandle);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hObject) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if let Some(file) = engine.registry().remove_file(hObject as usize) {
            drop(file); // release the stream/source Arc
            // Close the real carrier handle (its value == hObject) via the original.
            let close: PfnCloseHandle = orig_fn(Sym::CloseHandle);
            // SAFETY: `hObject` is our carrier handle; closing it exactly once.
            let ok = unsafe { close(hObject) };
            set_err(ERROR_SUCCESS);
            ok
        } else {
            pass()
        }
    })
}

// ---- Directory enumeration -------------------------------------------------

/// Build the matched entries for a virtual find pattern.
fn find_matches(engine: &crate::router::Engine, pattern_path: &Path) -> Vec<VfsDirEntry> {
    let leaf = pattern_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let parent = pattern_path.parent().unwrap_or(pattern_path);

    let mut out = Vec::new();
    if let Some(Ok(entries)) = engine.vfs().read_dir(parent) {
        for entry in entries {
            if wildcard_match(&leaf, &entry.name.to_string_lossy()) {
                out.push(entry);
            }
        }
    }
    // Exact-name lookup where the parent is not an enumerable directory node.
    if out.is_empty()
        && !leaf.contains(['*', '?'])
        && let Some(Ok(meta)) = engine.vfs().metadata(pattern_path)
    {
        out.push(VfsDirEntry {
            name: std::ffi::OsString::from(&leaf),
            metadata: meta,
        });
    }
    out
}

/// `FindFirstFileExW` shim.
pub(crate) extern "system" fn shim_FindFirstFileExW(
    lpFileName: PCWSTR,
    fInfoLevelId: i32,
    lpFindFileData: *mut c_void,
    fSearchOp: i32,
    lpSearchFilter: *const c_void,
    dwAdditionalFlags: u32,
) -> HANDLE {
    guard_abi(INVALID_HANDLE_VALUE, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnFindFirstFileExW = orig_fn(Sym::FindFirstFileExW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    lpFileName,
                    fInfoLevelId,
                    lpFindFileData,
                    fSearchOp,
                    lpSearchFilter,
                    dwAdditionalFlags,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide pattern (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_FILE_NOT_FOUND);
                INVALID_HANDLE_VALUE
            }
            Some((engine, Route::Virtual(path))) => {
                let mut matches = find_matches(&engine, &path);
                if matches.is_empty() {
                    set_err(ERROR_FILE_NOT_FOUND);
                    return INVALID_HANDLE_VALUE;
                }
                let Some(carrier) = carrier_find() else {
                    set_err(ERROR_INTERNAL_ERROR);
                    return INVALID_HANDLE_VALUE;
                };
                let first = matches.remove(0);
                // SAFETY: `lpFindFileData` is a caller-provided WIN32_FIND_DATAW.
                unsafe { fill_find_data(lpFindFileData.cast::<WIN32_FIND_DATAW>(), &first) };
                engine.registry().insert_find(
                    carrier as usize,
                    FindState {
                        entries: matches,
                        next: 0,
                    },
                );
                set_err(ERROR_SUCCESS);
                carrier
            }
        }
    })
}

/// `FindFirstFileW` shim (defaults to standard info level / name match).
pub(crate) extern "system" fn shim_FindFirstFileW(
    lpFileName: PCWSTR,
    lpFindFileData: *mut WIN32_FIND_DATAW,
) -> HANDLE {
    guard_abi(INVALID_HANDLE_VALUE, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnFindFirstFileW = orig_fn(Sym::FindFirstFileW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpFileName, lpFindFileData) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide pattern (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_FILE_NOT_FOUND);
                INVALID_HANDLE_VALUE
            }
            Some((engine, Route::Virtual(path))) => {
                let mut matches = find_matches(&engine, &path);
                if matches.is_empty() {
                    set_err(ERROR_FILE_NOT_FOUND);
                    return INVALID_HANDLE_VALUE;
                }
                let Some(carrier) = carrier_find() else {
                    set_err(ERROR_INTERNAL_ERROR);
                    return INVALID_HANDLE_VALUE;
                };
                let first = matches.remove(0);
                // SAFETY: caller-provided WIN32_FIND_DATAW.
                unsafe { fill_find_data(lpFindFileData, &first) };
                engine.registry().insert_find(
                    carrier as usize,
                    FindState {
                        entries: matches,
                        next: 0,
                    },
                );
                set_err(ERROR_SUCCESS);
                carrier
            }
        }
    })
}

/// `FindNextFileW` shim.
pub(crate) extern "system" fn shim_FindNextFileW(
    hFindFile: HANDLE,
    lpFindFileData: *mut WIN32_FIND_DATAW,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnFindNextFileW = orig_fn(Sym::FindNextFileW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFindFile, lpFindFileData) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        let Some(find) = engine.registry().get_find(hFindFile as usize) else {
            return pass();
        };
        let mut state = find
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let idx = state.next;
        if let Some(entry) = state.entries.get(idx) {
            // SAFETY: caller-provided WIN32_FIND_DATAW.
            unsafe { fill_find_data(lpFindFileData, entry) };
            state.next = idx + 1;
            set_err(ERROR_SUCCESS);
            1
        } else {
            set_err(ERROR_NO_MORE_FILES);
            0
        }
    })
}

/// `FindClose` shim.
pub(crate) extern "system" fn shim_FindClose(hFindFile: HANDLE) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnFindClose = orig_fn(Sym::FindClose);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(hFindFile) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        if engine.registry().remove_find(hFindFile as usize).is_some() {
            // Close the carrier find handle via the original.
            let close: PfnFindClose = orig_fn(Sym::FindClose);
            // SAFETY: `hFindFile` is our carrier find handle; closed exactly once.
            let ok = unsafe { close(hFindFile) };
            set_err(ERROR_SUCCESS);
            ok
        } else {
            pass()
        }
    })
}

// ---- Path attributes -------------------------------------------------------

/// `GetFileAttributesW` shim.
pub(crate) extern "system" fn shim_GetFileAttributesW(lpFileName: PCWSTR) -> u32 {
    guard_abi(INVALID_FILE_ATTRIBUTES, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFileAttributesW = orig_fn(Sym::GetFileAttributesW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpFileName) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_FILE_NOT_FOUND);
                INVALID_FILE_ATTRIBUTES
            }
            Some((engine, Route::Virtual(path))) => {
                if let Some(Ok(meta)) = engine.vfs().metadata(&path) {
                    set_err(ERROR_SUCCESS);
                    attrs_for(&meta)
                } else {
                    set_err(ERROR_FILE_NOT_FOUND);
                    INVALID_FILE_ATTRIBUTES
                }
            }
        }
    })
}

/// `GetFileAttributesExW` shim.
pub(crate) extern "system" fn shim_GetFileAttributesExW(
    lpFileName: PCWSTR,
    fInfoLevelId: i32,
    lpFileInformation: *mut c_void,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFileAttributesExW = orig_fn(Sym::GetFileAttributesExW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpFileName, fInfoLevelId, lpFileInformation) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_FILE_NOT_FOUND);
                0
            }
            Some((engine, Route::Virtual(path))) => {
                if let Some(Ok(meta)) = engine.vfs().metadata(&path) {
                    if lpFileInformation.is_null() {
                        set_err(ERROR_INVALID_PARAMETER);
                        return 0;
                    }
                    let (hi, lo) = split_hi_lo(meta.len);
                    // SAFETY: POD out struct, zeroed then initialized.
                    let mut data: WIN32_FILE_ATTRIBUTE_DATA = unsafe { core::mem::zeroed() };
                    data.dwFileAttributes = attrs_for(&meta);
                    data.nFileSizeHigh = hi;
                    data.nFileSizeLow = lo;
                    // SAFETY: caller-provided valid out pointer (level 0 == standard).
                    unsafe { *lpFileInformation.cast::<WIN32_FILE_ATTRIBUTE_DATA>() = data };
                    set_err(ERROR_SUCCESS);
                    1
                } else {
                    set_err(ERROR_FILE_NOT_FOUND);
                    0
                }
            }
        }
    })
}

// ---- Path resolution / cwd -------------------------------------------------

/// Write `text` (wide) into a caller buffer following the `GetFullPathNameW`
/// contract; return the length written (excluding NUL) or, if the buffer is too
/// small, the required length (including NUL). Also sets `lpFilePart`.
fn write_full_path_w(text: &str, buffer: PWSTR, buflen: u32, file_part: *mut PWSTR) -> u32 {
    let wide: Vec<u16> = text.encode_utf16().collect();
    let needed = wide.len();
    if buffer.is_null() || needed + 1 > buflen as usize {
        return u32::try_from(needed + 1).unwrap_or(u32::MAX);
    }
    // SAFETY: buffer has room for `needed + 1` wide chars (checked above).
    unsafe {
        for (i, unit) in wide.iter().enumerate() {
            *buffer.add(i) = *unit;
        }
        *buffer.add(needed) = 0;
    }
    if !file_part.is_null() {
        let prefix_units = text
            .rfind(['\\', '/'])
            .and_then(|idx| text.get(..=idx))
            .map_or(0, |p| p.encode_utf16().count());
        // SAFETY: `prefix_units <= needed`, so the pointer stays inside the buffer.
        unsafe { *file_part = buffer.add(prefix_units) };
    }
    set_err(ERROR_SUCCESS);
    u32::try_from(needed).unwrap_or(u32::MAX)
}

/// `GetFullPathNameW` shim.
pub(crate) extern "system" fn shim_GetFullPathNameW(
    lpFileName: PCWSTR,
    nBufferLength: u32,
    lpBuffer: PWSTR,
    lpFilePart: *mut PWSTR,
) -> u32 {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFullPathNameW = orig_fn(Sym::GetFullPathNameW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpFileName, nBufferLength, lpBuffer, lpFilePart) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_INVALID_NAME);
                0
            }
            Some((_, Route::Virtual(path))) => {
                write_full_path_w(&path.to_string_lossy(), lpBuffer, nBufferLength, lpFilePart)
            }
        }
    })
}

/// `GetFullPathNameA` shim.
pub(crate) extern "system" fn shim_GetFullPathNameA(
    lpFileName: PCSTR,
    nBufferLength: u32,
    lpBuffer: PSTR,
    lpFilePart: *mut PSTR,
) -> u32 {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetFullPathNameA = orig_fn(Sym::GetFullPathNameA);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpFileName, nBufferLength, lpBuffer, lpFilePart) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated byte path (or null).
        let Some(decoded) = (unsafe { decode_ansi(lpFileName) }) else {
            return pass();
        };
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_INVALID_NAME);
                0
            }
            Some((_, Route::Virtual(path))) => {
                let text = path.to_string_lossy();
                let bytes = text.as_bytes();
                let needed = bytes.len();
                if lpBuffer.is_null() || needed + 1 > nBufferLength as usize {
                    return u32::try_from(needed + 1).unwrap_or(u32::MAX);
                }
                // SAFETY: buffer has room for `needed + 1` bytes (checked above).
                unsafe {
                    for (i, b) in bytes.iter().enumerate() {
                        *lpBuffer.add(i) = *b;
                    }
                    *lpBuffer.add(needed) = 0;
                }
                if !lpFilePart.is_null() {
                    let prefix = text.rfind(['\\', '/']).map_or(0, |idx| idx + 1);
                    // SAFETY: `prefix <= needed`, still inside the buffer.
                    unsafe { *lpFilePart = lpBuffer.add(prefix) };
                }
                set_err(ERROR_SUCCESS);
                u32::try_from(needed).unwrap_or(u32::MAX)
            }
        }
    })
}

/// `GetCurrentDirectoryW` shim — returns the virtual cwd when one is tracked.
pub(crate) extern "system" fn shim_GetCurrentDirectoryW(
    nBufferLength: u32,
    lpBuffer: PWSTR,
) -> u32 {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetCurrentDirectoryW = orig_fn(Sym::GetCurrentDirectoryW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(nBufferLength, lpBuffer) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        let Some(engine) = current_engine() else {
            return pass();
        };
        match engine.virtual_cwd() {
            Some(cwd) => write_full_path_w(&cwd, lpBuffer, nBufferLength, core::ptr::null_mut()),
            None => pass(),
        }
    })
}

// ---- Mutations -------------------------------------------------------------

/// Translate a VFS mutation result into the Win32 `BOOL` + last-error convention.
fn mutation_result(result: std::io::Result<()>) -> BOOL {
    match result {
        Ok(()) => {
            set_err(ERROR_SUCCESS);
            1
        }
        Err(err) => {
            set_err(io_to_win(&err));
            0
        }
    }
}

/// `DeleteFileW` shim — remove a virtual file when writes are permitted; fail closed
/// (`ERROR_ACCESS_DENIED`) otherwise; passthrough for a real path.
pub(crate) extern "system" fn shim_DeleteFileW(lpFileName: PCWSTR) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnDeleteFileW = orig_fn(Sym::DeleteFileW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpFileName) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpFileName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_ACCESS_DENIED);
                0
            }
            Some((engine, Route::Virtual(path))) => {
                if !engine.allow_writes() {
                    set_err(ERROR_ACCESS_DENIED);
                    return 0;
                }
                if let Some(res) = engine.vfs().remove(&path) {
                    mutation_result(res)
                } else {
                    set_err(ERROR_FILE_NOT_FOUND);
                    0
                }
            }
        }
    })
}

/// `MoveFileExW` shim — rename within the virtual tree when the source is virtual and
/// writes are permitted; fail closed otherwise; passthrough for a real source.
pub(crate) extern "system" fn shim_MoveFileExW(
    lpExistingFileName: PCWSTR,
    lpNewFileName: PCWSTR,
    dwFlags: u32,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnMoveFileExW = orig_fn(Sym::MoveFileExW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpExistingFileName, lpNewFileName, dwFlags) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide paths (or null).
        let Some(from_wide) = (unsafe { decode_wide(lpExistingFileName) }) else {
            return pass();
        };
        let from_decoded = String::from_utf16_lossy(&from_wide);
        match route(&from_decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_ACCESS_DENIED);
                0
            }
            Some((engine, Route::Virtual(from))) => {
                if !engine.allow_writes() {
                    set_err(ERROR_ACCESS_DENIED);
                    return 0;
                }
                // SAFETY: caller's NUL-terminated wide destination (or null).
                let Some(to_wide) = (unsafe { decode_wide(lpNewFileName) }) else {
                    set_err(ERROR_INVALID_PARAMETER);
                    return 0;
                };
                let to_decoded = String::from_utf16_lossy(&to_wide);
                // Both endpoints must be virtual — a virtual→real move would leak the
                // synthetic file to disk, so it fails closed.
                match engine.classify(&to_decoded) {
                    Route::Virtual(to) => {
                        if let Some(res) = engine.vfs().rename(&from, &to) {
                            mutation_result(res)
                        } else {
                            set_err(ERROR_FILE_NOT_FOUND);
                            0
                        }
                    }
                    Route::Real | Route::Rejected => {
                        set_err(ERROR_ACCESS_DENIED);
                        0
                    }
                }
            }
        }
    })
}

/// `CreateDirectoryW` shim — create a virtual directory when writes are permitted;
/// fail closed otherwise; passthrough for a real path.
pub(crate) extern "system" fn shim_CreateDirectoryW(
    lpPathName: PCWSTR,
    lpSecurityAttributes: *const c_void,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnCreateDirectoryW = orig_fn(Sym::CreateDirectoryW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpPathName, lpSecurityAttributes) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpPathName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                set_err(ERROR_ACCESS_DENIED);
                0
            }
            Some((engine, Route::Virtual(path))) => {
                if !engine.allow_writes() {
                    set_err(ERROR_ACCESS_DENIED);
                    return 0;
                }
                if let Some(res) = engine.vfs().create_dir(&path) {
                    mutation_result(res)
                } else {
                    set_err(ERROR_ACCESS_DENIED);
                    0
                }
            }
        }
    })
}

// ---- Volume queries --------------------------------------------------------

/// `GetDiskFreeSpaceA` shim — synthetic answer only for a virtual root.
pub(crate) extern "system" fn shim_GetDiskFreeSpaceA(
    lpRootPathName: PCSTR,
    lpSectorsPerCluster: *mut u32,
    lpBytesPerSector: *mut u32,
    lpNumberOfFreeClusters: *mut u32,
    lpTotalNumberOfClusters: *mut u32,
) -> BOOL {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetDiskFreeSpaceA = orig_fn(Sym::GetDiskFreeSpaceA);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe {
                orig(
                    lpRootPathName,
                    lpSectorsPerCluster,
                    lpBytesPerSector,
                    lpNumberOfFreeClusters,
                    lpTotalNumberOfClusters,
                )
            }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated byte path (or null).
        let Some(decoded) = (unsafe { decode_ansi(lpRootPathName) }) else {
            return pass();
        };
        match route(&decoded) {
            Some((engine, Route::Virtual(_))) => {
                let write = |ptr: *mut u32, val: u32| {
                    if !ptr.is_null() {
                        // SAFETY: caller-provided out pointer, checked non-null.
                        unsafe { *ptr = val };
                    }
                };
                // Reflect the provider's configurable capacity when it reports one,
                // else a deterministic roomy default. Geometry: 512-byte sectors,
                let (sectors_per_cluster, total_clusters, free_clusters) =
                    match engine.vfs().volume_info() {
                        Some(v) => {
                            let bytes_per_cluster = u64::from(v.block_size.max(512));
                            let spc = u32::try_from(bytes_per_cluster / 512).unwrap_or(8);
                            let total =
                                u32::try_from(v.capacity / bytes_per_cluster).unwrap_or(u32::MAX);
                            let free =
                                u32::try_from(v.available / bytes_per_cluster).unwrap_or(u32::MAX);
                            (spc.max(1), total, free)
                        }
                        None => (8, 0x0100_0000, 0x0080_0000),
                    };
                write(lpSectorsPerCluster, sectors_per_cluster);
                write(lpBytesPerSector, 512);
                write(lpNumberOfFreeClusters, free_clusters);
                write(lpTotalNumberOfClusters, total_clusters);
                set_err(ERROR_SUCCESS);
                1
            }
            _ => pass(),
        }
    })
}

/// `GetDriveTypeW` shim — a virtual root is a fixed disk.
pub(crate) extern "system" fn shim_GetDriveTypeW(lpRootPathName: PCWSTR) -> u32 {
    guard_abi(0, PANIC_ERROR, || {
        let pass = || {
            let orig: PfnGetDriveTypeW = orig_fn(Sym::GetDriveTypeW);
            // SAFETY: forwarding the caller's exact arguments.
            unsafe { orig(lpRootPathName) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated wide path (or null).
        let Some(wide) = (unsafe { decode_wide(lpRootPathName) }) else {
            return pass();
        };
        let decoded = String::from_utf16_lossy(&wide);
        match route(&decoded) {
            Some((_, Route::Virtual(_))) => DRIVE_FIXED,
            _ => pass(),
        }
    })
}

// ---- Loader (auto_rescan) --------------------------------------------------

/// `LoadLibraryExW` shim — passthrough, then trigger a rescan for late modules.
pub(crate) extern "system" fn shim_LoadLibraryExW(
    lpLibFileName: PCWSTR,
    hFile: HANDLE,
    dwFlags: u32,
) -> HANDLE {
    guard_abi(core::ptr::null_mut(), PANIC_ERROR, || {
        let orig: PfnLoadLibraryExW = orig_fn(Sym::LoadLibraryExW);
        // SAFETY: forwarding the caller's exact arguments.
        let result = unsafe { orig(lpLibFileName, hFile, dwFlags) };
        if !result.is_null() {
            dispatch::trigger_rescan();
        }
        result
    })
}

/// `LoadLibraryA` shim — passthrough, then trigger a rescan for late modules.
pub(crate) extern "system" fn shim_LoadLibraryA(lpLibFileName: PCSTR) -> HANDLE {
    guard_abi(core::ptr::null_mut(), PANIC_ERROR, || {
        let orig: PfnLoadLibraryA = orig_fn(Sym::LoadLibraryA);
        // SAFETY: forwarding the caller's exact arguments.
        let result = unsafe { orig(lpLibFileName) };
        if !result.is_null() {
            dispatch::trigger_rescan();
        }
        result
    })
}

/// The address of the shim for `sym`, for building the replacement set. The
/// loader shims (`LoadLibrary*`) are included only when `auto_rescan` is enabled.
pub(crate) fn shim_address(sym: Sym) -> *const c_void {
    match sym {
        Sym::CreateFileW => shim_CreateFileW as *const c_void,
        Sym::CreateFileA => shim_CreateFileA as *const c_void,
        Sym::CreateFile2 => shim_CreateFile2 as *const c_void,
        Sym::ReadFile => shim_ReadFile as *const c_void,
        Sym::ReadFileEx => shim_ReadFileEx as *const c_void,
        Sym::WriteFile => shim_WriteFile as *const c_void,
        Sym::WriteFileEx => shim_WriteFileEx as *const c_void,
        Sym::SetFilePointer => shim_SetFilePointer as *const c_void,
        Sym::SetFilePointerEx => shim_SetFilePointerEx as *const c_void,
        Sym::GetFileSize => shim_GetFileSize as *const c_void,
        Sym::GetFileSizeEx => shim_GetFileSizeEx as *const c_void,
        Sym::GetFileType => shim_GetFileType as *const c_void,
        Sym::GetFileInformationByHandle => shim_GetFileInformationByHandle as *const c_void,
        Sym::GetFileInformationByHandleEx => shim_GetFileInformationByHandleEx as *const c_void,
        Sym::FlushFileBuffers => shim_FlushFileBuffers as *const c_void,
        Sym::SetEndOfFile => shim_SetEndOfFile as *const c_void,
        Sym::CloseHandle => shim_CloseHandle as *const c_void,
        Sym::FindFirstFileExW => shim_FindFirstFileExW as *const c_void,
        Sym::FindFirstFileW => shim_FindFirstFileW as *const c_void,
        Sym::FindNextFileW => shim_FindNextFileW as *const c_void,
        Sym::FindClose => shim_FindClose as *const c_void,
        Sym::GetFileAttributesW => shim_GetFileAttributesW as *const c_void,
        Sym::GetFileAttributesExW => shim_GetFileAttributesExW as *const c_void,
        Sym::GetFullPathNameW => shim_GetFullPathNameW as *const c_void,
        Sym::GetFullPathNameA => shim_GetFullPathNameA as *const c_void,
        Sym::GetCurrentDirectoryW => shim_GetCurrentDirectoryW as *const c_void,
        Sym::DeleteFileW => shim_DeleteFileW as *const c_void,
        Sym::MoveFileExW => shim_MoveFileExW as *const c_void,
        Sym::GetDiskFreeSpaceA => shim_GetDiskFreeSpaceA as *const c_void,
        Sym::GetDriveTypeW => shim_GetDriveTypeW as *const c_void,
        Sym::CreateDirectoryW => shim_CreateDirectoryW as *const c_void,
        Sym::LoadLibraryExW => shim_LoadLibraryExW as *const c_void,
        Sym::LoadLibraryA => shim_LoadLibraryA as *const c_void,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::namespace::Namespace;
    use crate::providers::MemoryFs;
    use crate::router::Engine;
    use crate::vfs::FileStream;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// `stable_file_id` is deterministic and case/separator-insensitive per path.
    #[test]
    fn stable_file_id_is_path_stable_and_distinct() {
        let a1 = stable_file_id(Path::new("C:\\__hookfs__\\id\\clip.braw"));
        let a2 = stable_file_id(Path::new("c:/__HOOKFS__/id/clip.braw")); // same path
        let b = stable_file_id(Path::new("C:\\__hookfs__\\id\\other.braw"));
        assert_eq!(
            a1, a2,
            "same virtual path (case/separator-insensitive) => same id"
        );
        assert_ne!(
            a1, b,
            "distinct paths => distinct id (with high probability)"
        );
        assert_ne!(a1, 0);
    }

    fn register(engine: &Engine, handle: usize, bytes: Vec<u8>, path: &str) {
        let stream: Box<dyn FileStream> = Box::new(Cursor::new(bytes));
        engine.registry().insert_file(
            handle,
            OpenFile {
                stream: OpenStream::Read(stream),
                path: PathBuf::from(path),
                writable: false,
                overlapped: false,
            },
        );
    }

    /// Register a **writable** virtual handle backed by a `MemoryFs` file, returning
    /// the engine so the caller can inspect the written bytes afterwards.
    fn register_writable(engine: &Engine, handle: usize, path: &Path) {
        let opts = OpenOptions::write_truncate();
        let stream = engine
            .vfs()
            .open_write(path, &opts)
            .expect("provider handles path")
            .expect("open_write");
        engine.registry().insert_file(
            handle,
            OpenFile {
                stream: OpenStream::Write(stream),
                path: path.to_owned(),
                writable: true,
                overlapped: false,
            },
        );
    }

    fn file_info(handle: usize) -> BY_HANDLE_FILE_INFORMATION {
        // SAFETY: POD out struct; the shim fully initializes it on success.
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { core::mem::zeroed() };
        let ok = shim_GetFileInformationByHandle(handle as HANDLE, &raw mut info);
        assert_eq!(
            ok, 1,
            "GetFileInformationByHandle should succeed for a virtual handle"
        );
        info
    }

    /// Item 4: two opens of the same virtual path report the same identity;
    /// distinct paths differ. Item 2: a 0-length read with a NULL buffer succeeds
    /// with 0 bytes and never forms a slice from null.
    #[test]
    fn same_path_stable_identity_and_zero_length_null_read() {
        let _serial = crate::dispatch::GLOBAL_STATE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let ns = Namespace::from_root(PathBuf::from("C:\\__hookfs__\\wtest"));
        let engine = Arc::new(Engine::new(MemoryFs::new(), ns, false));
        let id = crate::dispatch::try_activate(engine.clone()).expect("no active install in test");

        // Two independent opens of the SAME path, plus one of a different path.
        let (h1, h2, h3): (usize, usize, usize) = (0x1001, 0x1002, 0x2001);
        register(
            &engine,
            h1,
            vec![1, 2, 3, 4],
            "C:\\__hookfs__\\wtest\\clip.bin",
        );
        register(&engine, h2, vec![9, 9], "C:\\__hookfs__\\wtest\\clip.bin");
        register(&engine, h3, vec![0], "C:\\__hookfs__\\wtest\\other.bin");

        let i1 = file_info(h1);
        let i2 = file_info(h2);
        let i3 = file_info(h3);

        // Item 4: same virtual path => same file identity + volume serial.
        assert_eq!(
            (i1.nFileIndexHigh, i1.nFileIndexLow),
            (i2.nFileIndexHigh, i2.nFileIndexLow),
            "two opens of the same virtual path must report the same nFileIndex",
        );
        assert_eq!(i1.dwVolumeSerialNumber, i2.dwVolumeSerialNumber);
        assert_eq!(i1.dwVolumeSerialNumber, VIRTUAL_VOLUME_SERIAL);
        // Distinct paths => distinct identity (with high probability).
        assert_ne!(
            (i1.nFileIndexHigh, i1.nFileIndexLow),
            (i3.nFileIndexHigh, i3.nFileIndexLow),
        );

        // Item 2: 0-length read with a NULL buffer succeeds with 0 bytes, no UB.
        let mut got: u32 = 0xDEAD_BEEF;
        let ok = shim_ReadFile(
            h1 as HANDLE,
            core::ptr::null_mut(),
            0,
            &raw mut got,
            core::ptr::null_mut(),
        );
        assert_eq!(ok, 1, "a 0-length read must succeed");
        assert_eq!(got, 0, "a 0-length read must report 0 bytes read");

        engine.registry().remove_file(h1);
        engine.registry().remove_file(h2);
        engine.registry().remove_file(h3);
        crate::dispatch::deactivate(id);
    }

    /// `WriteFile` grows the file, `SetEndOfFile` truncates to the pointer, and a
    /// seek-to-start + `ReadFile` reads the written bytes back. Read-only handles
    /// still deny writes.
    #[test]
    fn writable_handle_write_truncate_read_back() {
        let _serial = crate::dispatch::GLOBAL_STATE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let ns = Namespace::from_root(PathBuf::from("C:\\__hookfs__\\wr"));
        let engine = Arc::new(Engine::new(MemoryFs::new(), ns, true));
        let id = crate::dispatch::try_activate(engine.clone()).expect("no active install in test");

        let path = Path::new("C:\\__hookfs__\\wr\\out.sidecar");
        let h: usize = 0x5001;
        register_writable(&engine, h, path);

        // WriteFile grows the file.
        let payload = b"sidecar-bytes-1234567890";
        let mut written: u32 = 0;
        let ok = shim_WriteFile(
            h as HANDLE,
            payload.as_ptr(),
            payload.len() as u32,
            &raw mut written,
            core::ptr::null_mut(),
        );
        assert_eq!(ok, 1, "WriteFile on a writable virtual handle must succeed");
        assert_eq!(written as usize, payload.len());

        // Size reflects the write.
        let mut size: i64 = 0;
        assert_eq!(shim_GetFileSizeEx(h as HANDLE, &raw mut size), 1);
        assert_eq!(size as usize, payload.len());

        // Seek to start and read the bytes back.
        let mut newpos: i64 = 0;
        assert_eq!(shim_SetFilePointerEx(h as HANDLE, 0, &raw mut newpos, 0), 1);
        let mut buf = vec![0u8; payload.len()];
        let mut read: u32 = 0;
        let ok = shim_ReadFile(
            h as HANDLE,
            buf.as_mut_ptr(),
            buf.len() as u32,
            &raw mut read,
            core::ptr::null_mut(),
        );
        assert_eq!(ok, 1);
        assert_eq!(&buf[..read as usize], payload);

        // Truncate at the current pointer (now == full length after the read): keep
        // it simple — seek to 5, SetEndOfFile, and confirm the new size.
        assert_eq!(shim_SetFilePointerEx(h as HANDLE, 5, &raw mut newpos, 0), 1);
        assert_eq!(shim_SetEndOfFile(h as HANDLE), 1);
        assert_eq!(shim_GetFileSizeEx(h as HANDLE, &raw mut size), 1);
        assert_eq!(size, 5);

        // A read-only handle denies writes with ACCESS_DENIED.
        let ro: usize = 0x5002;
        register(&engine, ro, vec![1, 2, 3], "C:\\__hookfs__\\wr\\ro.bin");
        let mut w2: u32 = 0;
        let denied = shim_WriteFile(
            ro as HANDLE,
            payload.as_ptr(),
            3,
            &raw mut w2,
            core::ptr::null_mut(),
        );
        assert_eq!(denied, 0, "writing a read-only virtual handle must fail");

        engine.registry().remove_file(h);
        engine.registry().remove_file(ro);
        crate::dispatch::deactivate(id);
    }
}
