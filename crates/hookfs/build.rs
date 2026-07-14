//! Build the engine test fixtures.
//!
//! The engine-level integration tests need a *real* loaded module that issues
//! file calls **through its import table** so `hookfs` can rebind those slots and
//! observe the redirect. On Windows this compiles `tests/fixtures/fs_target.rs`
//! into a self-contained `cdylib` (`HOOKFS_FIXTURE_DLL`); on **native** Linux it
//! compiles `tests/fixtures/fs_target_linux.rs` into a `cdylib`
//! (`HOOKFS_FIXTURE_SO`). Cross-compiling the Linux fixture from a non-Linux host
//! is skipped (no target linker), so the Linux engine test then skips at runtime.
//!
//! A build script reports failure by panicking, so `expect` is the correct idiom.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let host = std::env::var("HOST").unwrap_or_default();
    let target = std::env::var("TARGET").unwrap();

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());

    emit_backend_cfg(&target_os);

    if target_os == "windows" {
        let src = manifest_dir
            .join("tests")
            .join("fixtures")
            .join("fs_target.rs");
        println!("cargo:rerun-if-changed={}", src.display());
        // The `emit_backend_cfg` above is the only work needed for a dependency
        // build. This fixture backs the integration tests only; when the source
        // isn't present (consumed downstream, or published with `tests/fixtures/`
        // excluded) there's no test to run, so skip rather than shell out to
        // `rustc` or panic on the missing file. A dev checkout still builds it.
        if !src.exists() {
            return;
        }
        let output = out_dir.join("hookfs_fs_target.dll");
        let status = Command::new(&rustc)
            .args(["--edition", "2021", "--crate-type", "cdylib"])
            .args(["--crate-name", "hookfs_fs_target"])
            // Static CRT so the fixture DLL loads without an external UCRT/VCRUNTIME.
            .args(["-C", "target-feature=+crt-static"])
            .args(["--target", &target])
            .arg("-o")
            .arg(&output)
            .arg(&src)
            .status()
            .expect("failed to spawn rustc to build the hookfs test fixture");
        assert!(status.success(), "failed to build the hookfs test fixture");
        println!("cargo:rustc-env=HOOKFS_FIXTURE_DLL={}", output.display());
    } else if target_os == "linux" && host == target {
        // Only on a native Linux build: a cross build from another host lacks the
        // Linux target linker, and the Linux engine test skips without the env var.
        let src = manifest_dir
            .join("tests")
            .join("fixtures")
            .join("fs_target_linux.rs");
        println!("cargo:rerun-if-changed={}", src.display());
        // Test-only fixture — skip when the source isn't present (see the Windows
        // branch above); the `emit_backend_cfg` a dependency needs already ran.
        if !src.exists() {
            return;
        }
        let output = out_dir.join("libhookfs_fs_target.so");
        let status = Command::new(&rustc)
            .args(["--edition", "2021", "--crate-type", "cdylib"])
            .args(["--crate-name", "hookfs_fs_target"])
            .args(["--target", &target])
            .arg("-o")
            .arg(&output)
            .arg(&src)
            .status()
            .expect("failed to spawn rustc to build the Linux hookfs test fixture");
        assert!(
            status.success(),
            "failed to build the Linux hookfs test fixture"
        );
        println!("cargo:rustc-env=HOOKFS_FIXTURE_SO={}", output.display());
    }
}

/// Emit the `hookfs_backend` cfg for targets that have a `plthook` shim backend.
///
/// The whole routing/engine/registry/dispatch/shim stack is meaningful only where
/// `plthook` can install the shims — currently the Windows/PE (IAT), Linux/ELF
/// (GOT-PLT), and Darwin/Mach-O (macOS + iOS) engines. Every backend-only item is
/// gated with `#[cfg(hookfs_backend)]`
/// (and the platform-agnostic public surface — the VFS traits, providers,
/// `Options`/`Scope`, the error model, and the pure path helpers — stays available
/// everywhere). Non-backend targets (Android/bionic and the BSDs) compile the
/// `Error::UnsupportedPlatform` placeholder in `crate::install` instead
///
/// This is the single source of truth for "has a shim backend": to light the stack
/// up on a new platform, add its `CARGO_CFG_TARGET_OS` value to `BACKEND_TARGET_OS`
/// below — that one edit is all the platform-agnostic layer needs (the per-platform
/// shim/dispatch/`plthook` code is added separately, as new platform code).
fn emit_backend_cfg(target_os: &str) {
    // Keep in lockstep with the engines `plthook` ships: Windows/PE, Linux/ELF, and
    // code-signing / arm64e caveats are handled in the engine and documented, not
    const BACKEND_TARGET_OS: &[&str] = &["windows", "linux", "macos", "ios"];

    // Declare the custom cfg so `unexpected_cfgs` recognizes it (Rust �A 1.80).
    println!("cargo::rustc-check-cfg=cfg(hookfs_backend)");
    if BACKEND_TARGET_OS.contains(&target_os) {
        println!("cargo::rustc-cfg=hookfs_backend");
    }
}
