//! native `.so` fixture that issues libc file calls through its PLT/GOT, install
//! `hookfs` over a `MemoryFs`, and observe the redirect into the virtual
//! filesystem — cookie-backed `fopen`/`fread`, carrier-fd `open`/`read`/`lseek`,
//! `fseek`/`ftell`/`stat` size, non-virtual passthrough parity, and drop-restore.
//!
//! # Where this runs
//! This test runs on **native Linux (CI)**. On the Windows development host it is
//! cfg-compiled only, and even on a Linux *cross* build from another host the
//! fixture `.so` is not built (no target linker), so the test skips at runtime via
//! the absent `HOOKFS_FIXTURE_SO`. The full Blackmagic-RAW decode acceptance
//! (decode `sample.braw` from a `Cursor` through these ELF hooks) mirrors the
//! Linux, since `hookfs` must not depend on `braw-rs`.

#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use hookfs::{Hookfs, Options};
use std::ffi::{CString, c_char, c_int, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

type FixtureReadAll = unsafe extern "C" fn(*const c_char, *mut u8, u32) -> i64;
type FixtureReadAt = unsafe extern "C" fn(*const c_char, i64, *mut u8, u32) -> i64;
type FixtureSize = unsafe extern "C" fn(*const c_char) -> i64;
type FixtureCanOpen = unsafe extern "C" fn(*const c_char) -> c_int;
type FixtureWriteAll = unsafe extern "C" fn(*const c_char, *const u8, u32) -> i64;
type FixtureRemove = unsafe extern "C" fn(*const c_char) -> c_int;
type FixtureCount = unsafe extern "C" fn(*const c_char) -> c_int;

fn cpath(path: &Path) -> CString {
    CString::new(path.as_os_str().as_bytes()).expect("path with no interior NUL")
}

fn dlsym(handle: *mut c_void, name: &[u8]) -> usize {
    // SAFETY: `handle` is a live dlopen handle; `name` is NUL-terminated.
    let p = unsafe { libc::dlsym(handle, name.as_ptr().cast::<c_char>()) };
    p as usize
}

struct Fixture {
    handle: *mut c_void,
    anchor: usize,
    stdio_read_all: FixtureReadAll,
    fd_read_at: FixtureReadAt,
    stdio_size: FixtureSize,
    stat_size: FixtureSize,
    can_open: FixtureCanOpen,
    stdio_write_all: FixtureWriteAll,
    fd_write_all: FixtureWriteAll,
    remove: FixtureRemove,
    count_entries: FixtureCount,
}

impl Fixture {
    /// Load the fixture `.so`, or `None` when it was not built (skip the test).
    fn load() -> Option<Self> {
        let so = option_env!("HOOKFS_FIXTURE_SO")?;
        let c = CString::new(so).unwrap();
        // SAFETY: valid NUL-terminated path; RTLD_NOW resolves eagerly.
        let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW) };
        assert!(!handle.is_null(), "failed to dlopen fixture {so}");
        let anchor = dlsym(handle, b"fixture_stdio_read_all\0");
        assert_ne!(anchor, 0, "fixture_stdio_read_all not found");
        // SAFETY: the resolved symbols are the fixture's exported functions.
        Some(unsafe {
            Self {
                handle,
                anchor,
                stdio_read_all: std::mem::transmute::<usize, FixtureReadAll>(anchor),
                fd_read_at: std::mem::transmute::<usize, FixtureReadAt>(dlsym(
                    handle,
                    b"fixture_fd_read_at\0",
                )),
                stdio_size: std::mem::transmute::<usize, FixtureSize>(dlsym(
                    handle,
                    b"fixture_stdio_size\0",
                )),
                stat_size: std::mem::transmute::<usize, FixtureSize>(dlsym(
                    handle,
                    b"fixture_stat_size\0",
                )),
                can_open: std::mem::transmute::<usize, FixtureCanOpen>(dlsym(
                    handle,
                    b"fixture_can_open\0",
                )),
                stdio_write_all: std::mem::transmute::<usize, FixtureWriteAll>(dlsym(
                    handle,
                    b"fixture_stdio_write_all\0",
                )),
                fd_write_all: std::mem::transmute::<usize, FixtureWriteAll>(dlsym(
                    handle,
                    b"fixture_fd_write_all\0",
                )),
                remove: std::mem::transmute::<usize, FixtureRemove>(dlsym(
                    handle,
                    b"fixture_remove\0",
                )),
                count_entries: std::mem::transmute::<usize, FixtureCount>(dlsym(
                    handle,
                    b"fixture_count_entries\0",
                )),
            }
        })
    }

    fn anchor(&self) -> *const c_void {
        self.anchor as *const c_void
    }

    fn read_all(&self, path: &Path) -> Option<Vec<u8>> {
        let c = cpath(path);
        let mut buf = vec![0u8; 1 << 16];
        // SAFETY: valid path/buffer; the fixture reads at most `buf.len()` bytes.
        let n = unsafe { (self.stdio_read_all)(c.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
        (n >= 0).then(|| {
            buf.truncate(n as usize);
            buf
        })
    }

    fn read_at(&self, path: &Path, offset: i64, len: u32) -> Option<Vec<u8>> {
        let c = cpath(path);
        let mut buf = vec![0u8; len as usize];
        // SAFETY: valid path/buffer of `len`.
        let n = unsafe { (self.fd_read_at)(c.as_ptr(), offset, buf.as_mut_ptr(), len) };
        (n >= 0).then(|| {
            buf.truncate(n as usize);
            buf
        })
    }

    fn stdio_size(&self, path: &Path) -> i64 {
        let c = cpath(path);
        // SAFETY: valid path.
        unsafe { (self.stdio_size)(c.as_ptr()) }
    }

    fn stat_size(&self, path: &Path) -> i64 {
        let c = cpath(path);
        // SAFETY: valid path.
        unsafe { (self.stat_size)(c.as_ptr()) }
    }

    fn can_open(&self, path: &Path) -> bool {
        let c = cpath(path);
        // SAFETY: valid path.
        unsafe { (self.can_open)(c.as_ptr()) != 0 }
    }

    fn stdio_write_all(&self, path: &Path, data: &[u8]) -> i64 {
        let c = cpath(path);
        // SAFETY: valid path/data.
        unsafe { (self.stdio_write_all)(c.as_ptr(), data.as_ptr(), data.len() as u32) }
    }

    fn fd_write_all(&self, path: &Path, data: &[u8]) -> i64 {
        let c = cpath(path);
        // SAFETY: valid path/data.
        unsafe { (self.fd_write_all)(c.as_ptr(), data.as_ptr(), data.len() as u32) }
    }

    fn remove(&self, path: &Path) -> bool {
        let c = cpath(path);
        // SAFETY: valid path.
        unsafe { (self.remove)(c.as_ptr()) != 0 }
    }

    fn count_entries(&self, path: &Path) -> i32 {
        let c = cpath(path);
        // SAFETY: valid path.
        unsafe { (self.count_entries)(c.as_ptr()) }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: closing the handle we opened.
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

/// One sequential test exercising the whole engine surface. Kept as a single
/// `#[test]` because every installation shares the one process-global routing
/// engine, so the sections must not run in parallel.
#[test]
fn engine_end_to_end() {
    let Some(fixture) = Fixture::load() else {
        eprintln!("skipping: HOOKFS_FIXTURE_SO not set (not a native Linux build)");
        return;
    };

    // A payload big enough to require several fread/read chunks.
    let payload: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();

    // A real on-disk file to verify non-virtual passthrough parity.
    let real_path =
        std::env::temp_dir().join(format!("hookfs_passthru_{}.bin", std::process::id()));
    let real_bytes: Vec<u8> = (0..1234u32).map(|i| (i % 253) as u8).collect();
    std::fs::write(&real_path, &real_bytes).unwrap();

    let synthetic;
    {
        let hookfs = Hookfs::new();
        let _mount = hookfs
            .mount("data.bin", std::io::Cursor::new(payload.clone()))
            .unwrap();
        let _guard = hookfs
            .install(Options::for_module(fixture.anchor()))
            .unwrap();
        synthetic = hookfs.path_for("data.bin");

        // Redirect: the fixture's fopen/fread now read from memory (cookie stream).
        assert_eq!(
            fixture.read_all(&synthetic).as_deref(),
            Some(payload.as_slice())
        );

        // Size via the cookie fseek(SEEK_END)+ftell, and via the stat shim.
        assert_eq!(fixture.stdio_size(&synthetic), payload.len() as i64);
        assert_eq!(fixture.stat_size(&synthetic), payload.len() as i64);

        // Carrier-fd path: open/lseek/read a positioned slice; EOF is not an error.
        assert_eq!(
            fixture.read_at(&synthetic, 100, 50).as_deref(),
            Some(&payload[100..150])
        );
        assert_eq!(
            fixture
                .read_at(&synthetic, payload.len() as i64, 16)
                .as_deref(),
            Some(&[][..])
        );

        // Non-virtual passthrough parity: the real file still reads correctly.
        assert_eq!(
            fixture.read_all(&real_path).as_deref(),
            Some(real_bytes.as_slice())
        );
    }

    // After the guard drops, the synthetic path no longer resolves and never
    // touches disk (it does not exist), while the real file still reads.
    assert!(!synthetic.exists());
    assert!(
        !fixture.can_open(&synthetic),
        "synthetic path must fail once hooks are removed"
    );
    assert_eq!(
        fixture.read_all(&real_path).as_deref(),
        Some(real_bytes.as_slice())
    );

    std::fs::remove_file(&real_path).ok();

    writable_and_enumeration(&fixture);
    writes_denied_when_not_allowed(&fixture);
}

/// callback) and `open(O_WRONLY|O_CREAT|O_TRUNC)`+`write` (carrier fd) create virtual
/// files served entirely from memory, read back byte-for-byte, enumerate via
/// `opendir`/`readdir`, and delete via `remove` — none of it touching disk.
fn writable_and_enumeration(fixture: &Fixture) {
    let clip: Vec<u8> = (0..2000u32).map(|i| (i % 251) as u8).collect();
    let sidecar = b"{ \"iso\": 800 }".to_vec();
    let via_fd = b"trimmed-output-bytes".to_vec();

    let hookfs = Hookfs::new();
    let _clip = hookfs
        .mount("sample.braw", std::io::Cursor::new(clip))
        .unwrap();
    let _guard = hookfs
        .install(Options::for_module(fixture.anchor()).allow_writes(true))
        .unwrap();

    // Cookie write path: fopen("wb") + fwrite.
    let sidecar_path = hookfs.path_for("sample.sidecar");
    assert_eq!(
        fixture.stdio_write_all(&sidecar_path, &sidecar),
        sidecar.len() as i64
    );
    assert!(
        !sidecar_path.exists(),
        "the virtual sidecar must not exist on disk"
    );
    assert_eq!(
        hookfs.read_virtual("sample.sidecar").as_deref(),
        Some(sidecar.as_slice())
    );
    assert_eq!(
        fixture.read_all(&sidecar_path).as_deref(),
        Some(sidecar.as_slice())
    );
    assert_eq!(fixture.stat_size(&sidecar_path), sidecar.len() as i64);

    // Carrier-fd write path: open(O_WRONLY|O_CREAT|O_TRUNC) + write + ftruncate.
    let out_path = hookfs.path_for("trimmed.braw");
    assert_eq!(
        fixture.fd_write_all(&out_path, &via_fd),
        via_fd.len() as i64
    );
    assert_eq!(
        hookfs.read_virtual("trimmed.braw").as_deref(),
        Some(via_fd.as_slice())
    );

    // Directory enumeration returns the three siblings (clip + sidecar + output).
    // opendir yields our virtual listing (no "."/".." synthesized), so the count is
    // exactly the mounted entries.
    assert_eq!(
        fixture.count_entries(hookfs.virtual_dir()),
        3,
        "readdir must list all siblings"
    );

    // remove() deletes a virtual file.
    assert!(
        fixture.remove(&sidecar_path),
        "remove on a virtual file must succeed"
    );
    assert!(hookfs.read_virtual("sample.sidecar").is_none());
    assert_eq!(
        fixture.count_entries(hookfs.virtual_dir()),
        2,
        "after remove two remain"
    );
}

/// With writes disabled the read-only fail-closed behavior is preserved: `fopen("wb")`
/// on a virtual path fails, and nothing hits disk.
fn writes_denied_when_not_allowed(fixture: &Fixture) {
    let hookfs = Hookfs::new();
    let _clip = hookfs
        .mount("clip.braw", std::io::Cursor::new(vec![0u8; 16]))
        .unwrap();
    let _guard = hookfs
        .install(Options::for_module(fixture.anchor()))
        .unwrap();

    let denied = hookfs.path_for("denied.sidecar");
    assert_eq!(
        fixture.stdio_write_all(&denied, b"nope"),
        -1,
        "write must fail closed"
    );
    assert_eq!(
        fixture.fd_write_all(&denied, b"nope"),
        -1,
        "fd write must fail closed"
    );
    assert!(hookfs.read_virtual("denied.sidecar").is_none());
    assert!(!denied.exists());
}
