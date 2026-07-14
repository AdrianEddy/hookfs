//! Linux engine test fixture — compiled to a `cdylib` by `build.rs` on native
//! Linux.
//!
//! Each exported function issues libc file calls **through the module's PLT/GOT**
//! (`fopen`/`fread`/`fclose`, `open`/`read`/`lseek`/`close`, `fseek`/`ftell`,
//! `stat`), so `hookfs` can rebind those slots and a test can observe the redirect
//! into a virtual filesystem, then restore it. The stdio path deliberately drives
//! the `fopencookie`-backed `FILE*` (its `fread`/`fseek`/`ftell` are the intentional
//! cookie-stream passthroughs, §7.1).
#![crate_type = "cdylib"]
#![allow(clippy::missing_safety_doc)]

use core::ffi::{c_char, c_int, c_void};

type Ssize = isize;
type Size = usize;
type OffT = i64;

extern "C" {
    fn fopen(path: *const c_char, mode: *const c_char) -> *mut c_void;
    fn fread(ptr: *mut c_void, size: Size, nmemb: Size, stream: *mut c_void) -> Size;
    fn fwrite(ptr: *const c_void, size: Size, nmemb: Size, stream: *mut c_void) -> Size;
    fn fseek(stream: *mut c_void, offset: OffT, whence: c_int) -> c_int;
    fn ftell(stream: *mut c_void) -> OffT;
    fn fclose(stream: *mut c_void) -> c_int;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: Size) -> Ssize;
    fn write(fd: c_int, buf: *const c_void, count: Size) -> Ssize;
    fn ftruncate(fd: c_int, length: OffT) -> c_int;
    fn lseek(fd: c_int, offset: OffT, whence: c_int) -> OffT;
    fn close(fd: c_int) -> c_int;
    fn stat(path: *const c_char, buf: *mut c_void) -> c_int;
    fn remove(path: *const c_char) -> c_int;
    fn opendir(path: *const c_char) -> *mut c_void;
    fn readdir(dirp: *mut c_void) -> *mut c_void;
    fn closedir(dirp: *mut c_void) -> c_int;
}

const O_RDONLY: c_int = 0;
const O_WRONLY: c_int = 1;
const O_CREAT: c_int = 0o100;
const O_TRUNC: c_int = 0o1000;
const SEEK_SET: c_int = 0;
const SEEK_END: c_int = 2;

/// Read the whole file (up to `cap`) via stdio (`fopen`/`fread`/`fclose`). Returns
/// the byte count, or `-1`.
#[no_mangle]
pub unsafe extern "C" fn fixture_stdio_read_all(path: *const c_char, out: *mut u8, cap: u32) -> i64 {
    let mode = c"rb".as_ptr();
    let f = fopen(path, mode);
    if f.is_null() {
        return -1;
    }
    let cap = cap as usize;
    let mut total = 0usize;
    while total < cap {
        let n = fread(out.add(total).cast::<c_void>(), 1, cap - total, f);
        if n == 0 {
            break;
        }
        total += n;
    }
    fclose(f);
    total as i64
}

/// Report the file size via the cookie `fseek(SEEK_END)` + `ftell`.
#[no_mangle]
pub unsafe extern "C" fn fixture_stdio_size(path: *const c_char) -> i64 {
    let f = fopen(path, c"rb".as_ptr());
    if f.is_null() {
        return -1;
    }
    if fseek(f, 0, SEEK_END) != 0 {
        fclose(f);
        return -1;
    }
    let size = ftell(f);
    fclose(f);
    size
}

/// Read `len` bytes at `offset` via the fd path (`open`/`lseek`/`read`/`close`).
#[no_mangle]
pub unsafe extern "C" fn fixture_fd_read_at(
    path: *const c_char,
    offset: i64,
    out: *mut u8,
    len: u32,
) -> i64 {
    let fd = open(path, O_RDONLY);
    if fd < 0 {
        return -1;
    }
    if lseek(fd, offset, SEEK_SET) < 0 {
        close(fd);
        return -1;
    }
    let n = read(fd, out.cast::<c_void>(), len as usize);
    close(fd);
    if n < 0 { -1 } else { n as i64 }
}

/// Report the file size via `stat` (`st_size` at offset 48 on x86_64/aarch64).
#[no_mangle]
pub unsafe extern "C" fn fixture_stat_size(path: *const c_char) -> i64 {
    let mut buf = [0u8; 256];
    if stat(path, buf.as_mut_ptr().cast::<c_void>()) != 0 {
        return -1;
    }
    let mut sz = [0u8; 8];
    sz.copy_from_slice(&buf[48..56]);
    i64::from_ne_bytes(sz)
}

/// Try to open the path via stdio; `1` if it opened, `0` otherwise.
#[no_mangle]
pub unsafe extern "C" fn fixture_can_open(path: *const c_char) -> c_int {
    let f = fopen(path, c"rb".as_ptr());
    if f.is_null() {
        0
    } else {
        fclose(f);
        1
    }
}

/// Create/truncate `path` and write `len` bytes via stdio (`fopen("wb")`/`fwrite`/
/// `fclose`) — drives the `fopencookie` write callback. Returns bytes written or -1.
#[no_mangle]
pub unsafe extern "C" fn fixture_stdio_write_all(path: *const c_char, data: *const u8, len: u32) -> i64 {
    let f = fopen(path, c"wb".as_ptr());
    if f.is_null() {
        return -1;
    }
    let n = fwrite(data.cast::<c_void>(), 1, len as usize, f);
    let closed = fclose(f);
    if closed != 0 { -1 } else { n as i64 }
}

/// Create/truncate `path` and write `len` bytes via the fd path (`open`(O_WRONLY|
/// O_CREAT|O_TRUNC)/`write`/`ftruncate`/`close`). Returns bytes written or -1.
#[no_mangle]
pub unsafe extern "C" fn fixture_fd_write_all(path: *const c_char, data: *const u8, len: u32) -> i64 {
    let fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644);
    if fd < 0 {
        return -1;
    }
    let mut total = 0usize;
    while total < len as usize {
        let n = write(fd, data.add(total).cast::<c_void>(), len as usize - total);
        if n <= 0 {
            close(fd);
            return -1;
        }
        total += n as usize;
    }
    let t = ftruncate(fd, total as OffT);
    close(fd);
    if t != 0 { -1 } else { total as i64 }
}

/// Remove `path`. Returns `1` on success, `0` otherwise. Exercises `remove`.
#[no_mangle]
pub unsafe extern "C" fn fixture_remove(path: *const c_char) -> c_int {
    i32::from(remove(path) == 0)
}

/// Count entries in directory `path` via `opendir`/`readdir`/`closedir`. Returns the
/// entry count, or -1 if the directory could not be opened.
#[no_mangle]
pub unsafe extern "C" fn fixture_count_entries(path: *const c_char) -> c_int {
    let d = opendir(path);
    if d.is_null() {
        return -1;
    }
    let mut count = 0;
    while !readdir(d).is_null() {
        count += 1;
    }
    closedir(d);
    count
}
