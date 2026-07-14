//! that issues Win32 file calls through its IAT, install `hookfs`, and observe the
//! redirect into a virtual filesystem — then verify seek/EOF/size, non-virtual
//! passthrough parity, drop-restore, and panic safety at the ABI boundary.

#![cfg(windows)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use hookfs::vfs::{FileStream, OpenOptions, VfsDirEntry, VfsMetadata, VirtualFs};
use hookfs::{Hookfs, Options, Scope};
use std::ffi::c_void;
use std::io::{self, Cursor};
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::sync::Arc;

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

type FixtureReadAll = unsafe extern "system" fn(*const u16, *mut u8, u32) -> i64;
type FixtureReadAt = unsafe extern "system" fn(*const u16, i64, *mut u8, u32) -> i64;
type FixtureSize = unsafe extern "system" fn(*const u16) -> i64;
type FixtureCanOpen = unsafe extern "system" fn(*const u16) -> i32;
type FixtureWriteAll = unsafe extern "system" fn(*const u16, *const u8, u32) -> i64;
type FixtureDelete = unsafe extern "system" fn(*const u16) -> i32;
type FixtureCount = unsafe extern "system" fn(*const u16) -> i32;

fn wide(path: &Path) -> Vec<u16> {
    path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

fn proc_addr(module: HMODULE, name: &[u8]) -> usize {
    let p = unsafe { GetProcAddress(module, name.as_ptr()) };
    p.map_or(0, |f| f as usize)
}

struct Fixture {
    module: HMODULE,
    anchor: usize,
    read_all: FixtureReadAll,
    read_at: FixtureReadAt,
    size: FixtureSize,
    can_open: FixtureCanOpen,
    write_all: FixtureWriteAll,
    delete: FixtureDelete,
    count_entries: FixtureCount,
}

impl Fixture {
    fn load() -> Self {
        let dll = env!("HOOKFS_FIXTURE_DLL");
        let wpath = wide(Path::new(dll));
        let module = unsafe { LoadLibraryW(wpath.as_ptr()) };
        assert!(!module.is_null(), "failed to load fixture DLL {dll}");
        let anchor = proc_addr(module, b"fixture_read_all\0");
        assert_ne!(anchor, 0, "fixture_read_all not found");
        Self {
            module,
            anchor,
            read_all: unsafe { std::mem::transmute::<usize, FixtureReadAll>(anchor) },
            read_at: unsafe {
                std::mem::transmute::<usize, FixtureReadAt>(proc_addr(module, b"fixture_read_at\0"))
            },
            size: unsafe {
                std::mem::transmute::<usize, FixtureSize>(proc_addr(module, b"fixture_size\0"))
            },
            can_open: unsafe {
                std::mem::transmute::<usize, FixtureCanOpen>(proc_addr(
                    module,
                    b"fixture_can_open\0",
                ))
            },
            write_all: unsafe {
                std::mem::transmute::<usize, FixtureWriteAll>(proc_addr(
                    module,
                    b"fixture_write_all\0",
                ))
            },
            delete: unsafe {
                std::mem::transmute::<usize, FixtureDelete>(proc_addr(module, b"fixture_delete\0"))
            },
            count_entries: unsafe {
                std::mem::transmute::<usize, FixtureCount>(proc_addr(
                    module,
                    b"fixture_count_entries\0",
                ))
            },
        }
    }

    /// The address inside the fixture module used to scope the hooks.
    fn anchor(&self) -> *const c_void {
        self.anchor as *const c_void
    }

    fn read_all(&self, path: &Path) -> Option<Vec<u8>> {
        let w = wide(path);
        let mut buf = vec![0u8; 1 << 16];
        let n = unsafe { (self.read_all)(w.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
        if n < 0 {
            None
        } else {
            buf.truncate(n as usize);
            Some(buf)
        }
    }

    fn read_at(&self, path: &Path, offset: i64, len: u32) -> Option<Vec<u8>> {
        let w = wide(path);
        let mut buf = vec![0u8; len as usize];
        let n = unsafe { (self.read_at)(w.as_ptr(), offset, buf.as_mut_ptr(), len) };
        if n < 0 {
            None
        } else {
            buf.truncate(n as usize);
            Some(buf)
        }
    }

    fn size(&self, path: &Path) -> i64 {
        let w = wide(path);
        unsafe { (self.size)(w.as_ptr()) }
    }

    fn can_open(&self, path: &Path) -> bool {
        let w = wide(path);
        unsafe { (self.can_open)(w.as_ptr()) != 0 }
    }

    fn write_all(&self, path: &Path, data: &[u8]) -> i64 {
        let w = wide(path);
        unsafe { (self.write_all)(w.as_ptr(), data.as_ptr(), data.len() as u32) }
    }

    fn delete(&self, path: &Path) -> bool {
        let w = wide(path);
        unsafe { (self.delete)(w.as_ptr()) != 0 }
    }

    fn count_entries(&self, pattern: &Path) -> i32 {
        let w = wide(pattern);
        unsafe { (self.count_entries)(w.as_ptr()) }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        unsafe {
            FreeLibrary(self.module);
        }
    }
}

/// One sequential test exercising the whole engine surface. Kept as a single
/// `#[test]` because every installation shares the one process-global routing
/// engine, so the sections must not run in parallel.
#[test]
fn engine_end_to_end() {
    let fixture = Fixture::load();

    redirect_passthrough_seek_and_restore(&fixture);
    independent_cursors(&fixture);
    writable_create_write_readback_and_enumerate(&fixture);
    writes_denied_when_not_allowed(&fixture);
    provider_panic_is_contained(&fixture);
}

/// `WriteFile` + `SetEndOfFile` create a virtual file the VFS serves entirely from
/// memory (never on disk), it reads back byte-for-byte through the same path, the
/// directory enumerates both siblings, and `DeleteFileW` removes it.
fn writable_create_write_readback_and_enumerate(fixture: &Fixture) {
    let clip: Vec<u8> = (0..2000u32).map(|i| (i % 251) as u8).collect();
    let sidecar_bytes = b"{ \"iso\": 800, \"whiteBalance\": 5600 }".to_vec();

    let hookfs = Hookfs::new();
    let _clip_mount = hookfs
        .mount("sample.braw", Cursor::new(clip.clone()))
        .unwrap();
    let _guard = hookfs
        .install(Options::for_module(fixture.anchor()).allow_writes(true))
        .unwrap();

    let sidecar_path = hookfs.path_for("sample.sidecar");
    // The SDK-style writer creates + writes the sidecar entirely in the VFS.
    assert_eq!(
        fixture.write_all(&sidecar_path, &sidecar_bytes),
        sidecar_bytes.len() as i64,
        "virtual write must report all bytes written",
    );
    // Never hit disk.
    assert!(
        !sidecar_path.exists(),
        "the virtual sidecar must not exist on disk"
    );
    // Read back byte-for-byte, both through the provider and through the hooks.
    assert_eq!(
        hookfs.read_virtual("sample.sidecar").as_deref(),
        Some(sidecar_bytes.as_slice())
    );
    assert_eq!(
        fixture.read_all(&sidecar_path).as_deref(),
        Some(sidecar_bytes.as_slice())
    );
    assert_eq!(fixture.size(&sidecar_path), sidecar_bytes.len() as i64);

    // Directory enumeration returns both siblings (clip + sidecar).
    let glob = hookfs.virtual_dir().join("*");
    assert_eq!(
        fixture.count_entries(&glob),
        2,
        "enumeration must return clip + sidecar"
    );

    // Delete removes it from the VFS.
    assert!(
        fixture.delete(&sidecar_path),
        "DeleteFileW on a virtual file must succeed"
    );
    assert!(hookfs.read_virtual("sample.sidecar").is_none());
    assert_eq!(
        fixture.count_entries(&glob),
        1,
        "after delete only the clip remains"
    );
}

/// With writes disabled the exact read-only fail-closed behavior is preserved: a
/// `CreateFileW`(`GENERIC_WRITE`) on a virtual path fails, and nothing hits disk.
fn writes_denied_when_not_allowed(fixture: &Fixture) {
    let hookfs = Hookfs::new();
    let _clip_mount = hookfs
        .mount("clip.braw", Cursor::new(vec![0u8; 16]))
        .unwrap();
    let _guard = hookfs
        .install(Options::for_module(fixture.anchor()))
        .unwrap();

    let sidecar_path = hookfs.path_for("denied.sidecar");
    assert_eq!(
        fixture.write_all(&sidecar_path, b"nope"),
        -1,
        "a write must fail closed when allow_writes is not set",
    );
    assert!(hookfs.read_virtual("denied.sidecar").is_none());
    assert!(!sidecar_path.exists());
}

/// Redirect + size + seek/EOF + non-virtual passthrough parity + drop-restore.
fn redirect_passthrough_seek_and_restore(fixture: &Fixture) {
    // A payload big enough to require several ReadFile calls in the fixture loop.
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
            .mount("data.bin", Cursor::new(payload.clone()))
            .unwrap();
        let _guard = hookfs
            .install(Options::for_module(fixture.anchor()))
            .unwrap();
        synthetic = hookfs.path_for("data.bin");

        // Redirect: the fixture's CreateFileW/ReadFile now read from memory.
        assert_eq!(
            fixture.read_all(&synthetic).as_deref(),
            Some(payload.as_slice())
        );

        // Size via GetFileSizeEx.
        assert_eq!(fixture.size(&synthetic), payload.len() as i64);

        // Seek (SetFilePointerEx) + positioned read.
        assert_eq!(
            fixture.read_at(&synthetic, 100, 50).as_deref(),
            Some(&payload[100..150])
        );
        // Read at EOF yields zero bytes (not an error).
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
}

/// Independent opens of the same mounted source keep independent cursors.
fn independent_cursors(fixture: &Fixture) {
    let payload: Vec<u8> = (0..300u32).map(|i| (i % 97) as u8).collect();

    let hookfs = Hookfs::new();
    let _mount = hookfs
        .mount("clip.bin", Cursor::new(payload.clone()))
        .unwrap();
    let _guard = hookfs
        .install(Options::for_module(fixture.anchor()))
        .unwrap();
    let path = hookfs.path_for("clip.bin");

    // Each fixture call opens its own handle → its own cursor over the shared
    // source; positioned reads from different offsets are independent and correct.
    assert_eq!(
        fixture.read_at(&path, 10, 20).as_deref(),
        Some(&payload[10..30])
    );
    assert_eq!(
        fixture.read_at(&path, 200, 20).as_deref(),
        Some(&payload[200..220])
    );
    assert_eq!(fixture.read_all(&path).as_deref(), Some(payload.as_slice()));
}

/// A provider that panics inside `open` must not unwind across the FFI boundary:
/// the shim's `catch_unwind` turns it into a native failure (R15).
fn provider_panic_is_contained(fixture: &Fixture) {
    struct PanicFs;
    impl VirtualFs for PanicFs {
        fn open(&self, _p: &Path, _o: &OpenOptions) -> Option<io::Result<Box<dyn FileStream>>> {
            panic!("provider deliberately panics");
        }
        fn metadata(&self, _p: &Path) -> Option<io::Result<VfsMetadata>> {
            panic!("provider deliberately panics");
        }
        fn read_dir(&self, _p: &Path) -> Option<io::Result<Vec<VfsDirEntry>>> {
            None
        }
    }

    let prefix = Path::new("C:\\__hookfs_panic_test__");
    let _guard = hookfs::install(
        Arc::new(PanicFs),
        Options::for_prefix(prefix).scope(Scope::Module(fixture.anchor() as usize)),
    )
    .unwrap();

    // Opening a virtual path drives the panicking provider; the process must
    // survive and the open must fail cleanly.
    let path = prefix.join("boom.bin");
    assert!(
        !fixture.can_open(&path),
        "panicking open must fail, not crash"
    );
}
