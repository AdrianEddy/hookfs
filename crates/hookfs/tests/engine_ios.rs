//! iOS/iPadOS `hookfs` integration test.
//!
//! The test installs the Darwin shims into this binary, then verifies virtual-file
//! access through stdio, descriptor, and metadata APIs. It runs when built for an
//! iOS target; portable VFS behavior is covered by the host test suites.

#![cfg(target_os = "ios")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use core::ffi::c_void;
use hookfs::{Hookfs, Options};
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// An address inside this test binary, used to acquire its image for `Scope::Module`.
extern "C" fn anchor() {}

fn cpath(path: &Path) -> CString {
    CString::new(path.as_os_str().as_bytes()).expect("path with no interior NUL")
}

/// Read a whole file through libc stdio (`fopen`/`fread`/`fclose`) A?€�t the cookie-stream
/// path for a virtual clip. `None` if `fopen` fails (e.g. the synthetic path once the
/// hooks are gone).
fn stdio_read_all(path: &Path) -> Option<Vec<u8>> {
    let c = cpath(path);
    // SAFETY: valid NUL-terminated path and mode string.
    let file = unsafe { libc::fopen(c.as_ptr(), c"rb".as_ptr()) };
    if file.is_null() {
        return None;
    }
    let mut out = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        // SAFETY: `chunk` is a valid buffer of `chunk.len()` bytes; `file` is live.
        let n = unsafe { libc::fread(chunk.as_mut_ptr().cast::<c_void>(), 1, chunk.len(), file) };
        if n == 0 {
            break;
        }
        out.extend_from_slice(&chunk[..n]);
    }
    // SAFETY: closing the FILE* we opened exactly once.
    unsafe { libc::fclose(file) };
    Some(out)
}

/// Read a positioned slice through the carrier-fd path (`open`/`lseek`/`read`/`close`).
fn fd_read_at(path: &Path, offset: i64, len: usize) -> Option<Vec<u8>> {
    let c = cpath(path);
    // SAFETY: valid NUL-terminated path; O_RDONLY takes no mode argument.
    let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY) };
    if fd < 0 {
        return None;
    }
    // SAFETY: `fd` is a live descriptor.
    let seeked = unsafe { libc::lseek(fd, offset, libc::SEEK_SET) };
    let mut buf = vec![0u8; len];
    // SAFETY: `buf` addresses `len` writable bytes; `fd` is live.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast::<c_void>(), len) };
    // SAFETY: closing the descriptor we opened exactly once.
    unsafe { libc::close(fd) };
    if seeked < 0 || n < 0 {
        return None;
    }
    buf.truncate(n as usize);
    Some(buf)
}

/// The size reported by `stat` for a path (`-1` on failure).
fn stat_size(path: &Path) -> i64 {
    let c = cpath(path);
    // SAFETY: `st` is POD; `stat` fills it. `c`/`st` are valid.
    let mut st: libc::stat = unsafe { core::mem::zeroed() };
    let rc = unsafe { libc::stat(c.as_ptr(), &raw mut st) };
    if rc == 0 { st.st_size as i64 } else { -1 }
}

/// One sequential test exercising the whole engine surface. Kept as a single `#[test]`
/// because every installation shares the one process-global routing engine, so the
/// sections must not run in parallel.
#[test]
fn engine_end_to_end() {
    // A payload big enough to require several fread/read chunks.
    let payload: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();

    // A real on-disk file to verify non-virtual passthrough parity.
    let real_path =
        std::env::temp_dir().join(format!("hookfs_ios_passthru_{}.bin", std::process::id()));
    let real_bytes: Vec<u8> = (0..1234u32).map(|i| (i % 253) as u8).collect();
    std::fs::write(&real_path, &real_bytes).unwrap();

    let synthetic;
    {
        let hookfs = Hookfs::new();
        let _mount = hookfs
            .mount("clip.braw", std::io::Cursor::new(payload.clone()))
            .unwrap();
        // Hook this test binary's own image (Scope::Module by an address inside it).
        let _guard = hookfs
            .install(Options::for_module(anchor as *const c_void))
            .unwrap();
        synthetic = hookfs.path_for("clip.braw");

        // Cookie-stream path: fopen/fread now read from memory via `funopen`.
        assert_eq!(
            stdio_read_all(&synthetic).as_deref(),
            Some(payload.as_slice())
        );

        // The `stat` shim reports the virtual size.
        assert_eq!(stat_size(&synthetic), payload.len() as i64);

        // Carrier-fd path: open/lseek/read a positioned slice; EOF is not an error.
        assert_eq!(
            fd_read_at(&synthetic, 100, 50).as_deref(),
            Some(&payload[100..150])
        );
        assert_eq!(
            fd_read_at(&synthetic, payload.len() as i64, 16).as_deref(),
            Some(&[][..])
        );

        // Non-virtual passthrough parity: the real file still reads correctly.
        assert_eq!(
            stdio_read_all(&real_path).as_deref(),
            Some(real_bytes.as_slice())
        );
    }

    // After the guard drops, the synthetic path no longer resolves and never touches
    // disk (it does not exist), while the real file still reads.
    assert!(!synthetic.exists());
    assert!(
        stdio_read_all(&synthetic).is_none(),
        "synthetic path must fail once hooks are removed"
    );
    assert_eq!(
        stdio_read_all(&real_path).as_deref(),
        Some(real_bytes.as_slice())
    );

    std::fs::remove_file(&real_path).ok();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//
// The engine test above is self-contained and needs no external binary. The FULL
// BRAW acceptance A?€�t decoding `sample.braw` from a `Cursor` through the framework's
// own file I/O A?€�t additionally requires the `BlackmagicRawAPI.framework` **device
// binary**, which is NOT in this checkout (`braw-rs/sdk/iPadOS` ships only
// Include/Samples). It therefore cannot be written as a concrete test here; it must
// be run on a physical iPad against the embedded framework, following this procedure
//
//   1. Embed `BlackmagicRawAPI.framework` in an app bundle; load the factory and
//      discover the SDK image by the address of `CreateBlackmagicRawFactoryInstance`
//      (NOT a basename).
//   2. `hookfs::install(Options::for_module(factory_addr))` BEFORE the first
//      `OpenClip`, then mount the clip bytes under a synthetic path.
//   3. Decode `sample.braw` from a `Cursor<Vec<u8>>` while the physical file is made
//      unavailable; open ONLY the synthetic path.
//   4. Compare width/height/frame count/rate, clip + frame metadata, timecodes, and
//      decoded-frame hashes against decoding the real file.
//   5. Read frames sequential / reverse / random / repeated / concurrent.
//   6. Sidecar absent / valid / virtual-sibling.
//   7. Load the CPU/GPU decoder plugins (auto_rescan); assert no carrier fd/handle
//      escapes to an unknown consumer, and the trace contains only known hooks fo
//      virtual-prefix paths.
//   8. Confirm the framework is plain arm64 (NOT arm64e) A?€�t else `install` refuses its
//      authenticated slots with `Error::AuthenticatedSlot` (R1), and arm64e support
//      remains out of scope until PAC-aware handling is implemented and device-tested.
//
// Release gate: this matrix, plus the `funopen` availability and `mach_vm_protect`
// const-page behavior, MUST pass on physical hardware before any iPadOS support is
// claimed. App Store distribution of runtime import rebinding has review implications
