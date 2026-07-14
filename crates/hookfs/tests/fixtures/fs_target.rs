//! Engine test fixture — compiled to a `cdylib` by `build.rs`.
//!
//! Each exported function issues Win32 file calls **through the module's IAT**
//! (`CreateFileW`/`GetFileType`/`GetFileSizeEx`/`SetFilePointerEx`/`ReadFile`/
//! `CloseHandle`), so `hookfs` can rebind those slots and a test can observe the
//! redirect into a virtual filesystem, then restore it. Built with edition 2021
//! and a static CRT so the DLL is self-contained.
#![crate_type = "cdylib"]
#![allow(clippy::missing_safety_doc)]

use core::ffi::c_void;

type Handle = *mut c_void;

/// `WIN32_FIND_DATAW` — enough to read back `cFileName` (offset 44).
#[repr(C)]
struct FindDataW {
    dw_file_attributes: u32,
    ft_creation_time: [u32; 2],
    ft_last_access_time: [u32; 2],
    ft_last_write_time: [u32; 2],
    n_file_size_high: u32,
    n_file_size_low: u32,
    dw_reserved0: u32,
    dw_reserved1: u32,
    c_file_name: [u16; 260],
    c_alternate_file_name: [u16; 14],
}

#[link(name = "kernel32")]
extern "system" {
    fn CreateFileW(
        name: *const u16,
        access: u32,
        share: u32,
        sa: *const c_void,
        disposition: u32,
        flags: u32,
        template: Handle,
    ) -> Handle;
    fn ReadFile(h: Handle, buf: *mut u8, to_read: u32, read: *mut u32, ov: *mut c_void) -> i32;
    fn WriteFile(h: Handle, buf: *const u8, to_write: u32, written: *mut u32, ov: *mut c_void) -> i32;
    fn SetEndOfFile(h: Handle) -> i32;
    fn FlushFileBuffers(h: Handle) -> i32;
    fn GetFileSizeEx(h: Handle, size: *mut i64) -> i32;
    fn SetFilePointerEx(h: Handle, distance: i64, new_pos: *mut i64, method: u32) -> i32;
    fn GetFileType(h: Handle) -> u32;
    fn CloseHandle(h: Handle) -> i32;
    fn DeleteFileW(name: *const u16) -> i32;
    fn FindFirstFileW(name: *const u16, data: *mut FindDataW) -> Handle;
    fn FindNextFileW(h: Handle, data: *mut FindDataW) -> i32;
    fn FindClose(h: Handle) -> i32;
}

const GENERIC_READ: u32 = 0x8000_0000;
const GENERIC_WRITE: u32 = 0x4000_0000;
const FILE_SHARE_READ: u32 = 1;
const OPEN_EXISTING: u32 = 3;
const CREATE_ALWAYS: u32 = 2;
const INVALID_HANDLE: isize = -1;

unsafe fn open(path: *const u16) -> Handle {
    CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        core::ptr::null(),
        OPEN_EXISTING,
        0,
        core::ptr::null_mut(),
    )
}

fn is_invalid(h: Handle) -> bool {
    h as isize == INVALID_HANDLE || h.is_null()
}

/// Read the whole file (up to `cap`) into `out`. Returns bytes read, or `-1`.
/// Exercises `CreateFileW` + `GetFileType` + `ReadFile` (loop to EOF) + `CloseHandle`.
#[no_mangle]
pub extern "system" fn fixture_read_all(path: *const u16, out: *mut u8, cap: u32) -> i64 {
    unsafe {
        let h = open(path);
        if is_invalid(h) {
            return -1;
        }
        let _ = GetFileType(h);
        let mut total: u32 = 0;
        while total < cap {
            let mut read: u32 = 0;
            let ok = ReadFile(h, out.add(total as usize), cap - total, &mut read, core::ptr::null_mut());
            if ok == 0 {
                CloseHandle(h);
                return -1;
            }
            if read == 0 {
                break; // EOF — not an error.
            }
            total += read;
        }
        CloseHandle(h);
        total as i64
    }
}

/// Read `len` bytes starting at `offset` into `out`. Returns bytes read, or `-1`.
/// Exercises `SetFilePointerEx` (FILE_BEGIN) + a single `ReadFile`.
#[no_mangle]
pub extern "system" fn fixture_read_at(path: *const u16, offset: i64, out: *mut u8, len: u32) -> i64 {
    unsafe {
        let h = open(path);
        if is_invalid(h) {
            return -1;
        }
        let mut new_pos: i64 = 0;
        if SetFilePointerEx(h, offset, &mut new_pos, 0) == 0 {
            CloseHandle(h);
            return -1;
        }
        let mut read: u32 = 0;
        let ok = ReadFile(h, out, len, &mut read, core::ptr::null_mut());
        CloseHandle(h);
        if ok == 0 { -1 } else { read as i64 }
    }
}

/// Return the file size via `GetFileSizeEx`, or `-1`.
#[no_mangle]
pub extern "system" fn fixture_size(path: *const u16) -> i64 {
    unsafe {
        let h = open(path);
        if is_invalid(h) {
            return -1;
        }
        let mut size: i64 = 0;
        let ok = GetFileSizeEx(h, &mut size);
        CloseHandle(h);
        if ok == 0 { -1 } else { size }
    }
}

/// Just try to open the path; returns `1` if it opened, `0` otherwise.
#[no_mangle]
pub extern "system" fn fixture_can_open(path: *const u16) -> i32 {
    unsafe {
        let h = open(path);
        if is_invalid(h) {
            0
        } else {
            CloseHandle(h);
            1
        }
    }
}

/// Create/truncate `path` and write `len` bytes from `data`, then `SetEndOfFile`
/// and flush. Exercises `CreateFileW` (GENERIC_WRITE/CREATE_ALWAYS) + `WriteFile`
/// (loop) + `SetEndOfFile` + `FlushFileBuffers` + `CloseHandle`. Returns bytes
/// written, or `-1`.
#[no_mangle]
pub extern "system" fn fixture_write_all(path: *const u16, data: *const u8, len: u32) -> i64 {
    unsafe {
        let h = CreateFileW(
            path,
            GENERIC_WRITE,
            0,
            core::ptr::null(),
            CREATE_ALWAYS,
            0,
            core::ptr::null_mut(),
        );
        if is_invalid(h) {
            return -1;
        }
        let mut total: u32 = 0;
        while total < len {
            let mut written: u32 = 0;
            let ok = WriteFile(h, data.add(total as usize), len - total, &mut written, core::ptr::null_mut());
            if ok == 0 {
                CloseHandle(h);
                return -1;
            }
            if written == 0 {
                break;
            }
            total += written;
        }
        let end = SetEndOfFile(h);
        let flushed = FlushFileBuffers(h);
        CloseHandle(h);
        if end == 0 || flushed == 0 {
            -1
        } else {
            total as i64
        }
    }
}

/// Delete `path`. Returns `1` on success, `0` on failure. Exercises `DeleteFileW`.
#[no_mangle]
pub extern "system" fn fixture_delete(path: *const u16) -> i32 {
    unsafe { i32::from(DeleteFileW(path) != 0) }
}

/// Enumerate the directory glob `pattern` and return the number of entries found.
/// Exercises `FindFirstFileW` + `FindNextFileW` + `FindClose`. Returns `-1` if the
/// first find failed.
#[no_mangle]
pub extern "system" fn fixture_count_entries(pattern: *const u16) -> i32 {
    unsafe {
        let mut data: FindDataW = core::mem::zeroed();
        let h = FindFirstFileW(pattern, &mut data);
        if is_invalid(h) {
            return -1;
        }
        let mut count = 1;
        while FindNextFileW(h, &mut data) != 0 {
            count += 1;
        }
        FindClose(h);
        count
    }
}
