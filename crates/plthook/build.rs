//! Build the engine test fixtures.
//!
//! The integration tests need a *real* loaded module whose IAT the engine can
//! rebind. On Windows this script compiles `tests/fixtures/hook_target.rs` into
//! two self-contained `cdylib`s (two distinct modules → two slots for the same
//! symbol, exercising the multi-module path) and exposes their paths to the
//! tests via `PLTHOOK_FIXTURE_DLL` / `PLTHOOK_FIXTURE_DLL_B`.
//!
//! Compiling a `cdylib` directly with `rustc` (rather than a workspace member +
//! unstable artifact-dependency) keeps the fixture build self-contained on stable
//! Rust. A build script reports failure by panicking, so `expect` is the correct
//! idiom here.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // The fixtures back the Windows/PE engine tests only.
    if target_os != "windows" {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let fixtures = manifest_dir.join("tests").join("fixtures");
    let src = fixtures.join("hook_target.rs");
    let delay_src = fixtures.join("delay_target.rs");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", delay_src.display());

    // These fixtures back the integration tests only, but a build script runs on
    // *every* build — including when `plthook` is a transitive dependency. When the
    // crate is consumed downstream (or published with `tests/fixtures/` excluded)
    // the source isn't present: no test can run, so skip silently rather than
    // shelling out to `rustc` for a `cdylib` nobody loads (or panicking on the
    // missing source). In a dev checkout the fixtures exist, so tests still build.
    if !src.exists() {
        return;
    }

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let target = std::env::var("TARGET").unwrap();

    // Compile `source` into a self-contained `cdylib`, appending `extra` rustc
    // args. Returns `Ok(dll)` or `Err(())` so the delay-load fixture can be
    // skipped (rather than failing every test) on a toolchain that cannot link
    // `/DELAYLOAD` + `delayimp.lib`.
    let build = |crate_name: &str, source: &Path, extra: &[&str]| -> Result<PathBuf, ()> {
        let output = out_dir.join(format!("{crate_name}.dll"));
        let status = Command::new(&rustc)
            .args(["--edition", "2021"])
            .args(["--crate-type", "cdylib"])
            .args(["--crate-name", crate_name])
            // Self-contained: statically link the CRT so the DLL loads without an
            // external VCRUNTIME/UCRT dependency.
            .args(["-C", "target-feature=+crt-static"])
            .args(["--target", &target])
            .args(extra)
            .arg("-o")
            .arg(&output)
            .arg(source)
            .status()
            .expect("failed to spawn rustc to build the test fixture");
        if status.success() {
            Ok(output)
        } else {
            Err(())
        }
    };

    let primary =
        build("plthook_hook_target", &src, &[]).expect("build fixture `plthook_hook_target`");
    let secondary =
        build("plthook_hook_target_b", &src, &[]).expect("build fixture `plthook_hook_target_b`");

    println!("cargo:rustc-env=PLTHOOK_FIXTURE_DLL={}", primary.display());
    println!(
        "cargo:rustc-env=PLTHOOK_FIXTURE_DLL_B={}",
        secondary.display()
    );

    // Delay-load fixture: `winmm.dll` is delay-imported, so its delay IAT slot
    // holds the `__delayLoadHelper2` stub until first call — the exact shape the
    // engine's delay-original resolution must handle. `delayimp.lib` ships beside
    // the standard MSVC libraries rustc already links, so this normally succeeds;
    // if it can't be linked on some toolchain, the fixture is skipped and the
    // corresponding test (which gates on `option_env!`) is skipped too, rather
    // than breaking the whole suite.
    match build(
        "plthook_delay_target",
        &delay_src,
        &[
            "-C",
            "link-arg=/DELAYLOAD:winmm.dll",
            "-C",
            "link-arg=delayimp.lib",
        ],
    ) {
        Ok(delay) => {
            println!(
                "cargo:rustc-env=PLTHOOK_DELAY_FIXTURE_DLL={}",
                delay.display()
            );
        }
        Err(()) => {
            println!(
                "cargo:warning=could not build the /DELAYLOAD fixture (delayimp.lib \
                 unavailable?); the delay-import engine test will be skipped"
            );
        }
    }
}
