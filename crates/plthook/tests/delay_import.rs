//! Live `/DELAYLOAD` engine test: enumerate → install → call-through →
//! passthrough via `original()` → uninstall, on a genuine delay-load IAT slot
//! whose providing DLL (`winmm.dll`) is not yet mapped at install time.
//!
//! This is the delay-import counterpart to `engine.rs`. It pins down the fix for
//! the delay-original defect: the engine must resolve a delay import's *real*
//! callee authoritatively (force-loading the provider) rather than hand out the
//! `__delayLoadHelper2` stub the slot holds before first call. Passing that stub
//! through would run the loader helper, which rewrites the very slot we patched
//! and silently drops the hook — so the decisive assertion is that our hook
//! *survives* a passthrough call.
//!
//! The fixture is built by `build.rs`; if that toolchain cannot link
//! `/DELAYLOAD` the env var is unset and this test skips (rather than failing).
#![cfg(windows)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_transmute_annotations
)]

use core::ffi::c_void;
use plthook::{ImportKind, Module, Replacement, Symbol, install};
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};

/// A `call_timegettime`-shaped export.
type ZeroArgU32 = unsafe extern "system" fn() -> u32;

/// Sentinel returned by the replacement — distinct from any real `timeGetTime`.
extern "system" fn fake_time() -> u32 {
    0xDEAD_BEEF
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// `GetModuleHandleW` as a bare address (0 if the module is not mapped).
fn module_handle(name: &str) -> usize {
    // SAFETY: `wide(name)` is a valid NUL-terminated UTF-16 string.
    unsafe { GetModuleHandleW(wide(name).as_ptr()) as usize }
}

/// Resolve an export and transmute it to a callable `extern "system" fn() -> u32`.
fn export(handle: HMODULE, name: &str) -> ZeroArgU32 {
    let mut c_name = name.as_bytes().to_vec();
    c_name.push(0);
    // SAFETY: `c_name` is a valid NUL-terminated symbol name; `handle` is live.
    let proc = unsafe { GetProcAddress(handle, c_name.as_ptr()) };
    let proc = proc.unwrap_or_else(|| panic!("export `{name}` not found"));
    // SAFETY: the fixture declares this export as `extern "system" fn() -> u32`.
    unsafe { core::mem::transmute::<_, ZeroArgU32>(proc) }
}

fn call(f: ZeroArgU32) -> u32 {
    // SAFETY: `f` is a valid zero-argument `extern "system"` function.
    unsafe { f() }
}

#[test]
fn delay_import_install_passthrough_restore() {
    let Some(path) = option_env!("PLTHOOK_DELAY_FIXTURE_DLL") else {
        eprintln!("skipping: /DELAYLOAD fixture was not built on this toolchain");
        return;
    };

    // Load the delay fixture. Delay-load defers `winmm.dll` to first call, so
    // loading the fixture does NOT map winmm — its delay slot holds the load stub.
    // SAFETY: `wide(path)` is a valid NUL-terminated path.
    let handle = unsafe { LoadLibraryW(wide(path).as_ptr()) };
    assert!(!handle.is_null(), "LoadLibraryW failed for {path}");

    // Precondition for a meaningful regression test: winmm must be unmapped so the
    // engine has to FORCE-LOAD it to resolve the original. If winmm were already
    // loaded, even the old (buggy) `resolve_export` would succeed and the bug
    // would not reproduce — so we require the unloaded state and fail loudly
    // otherwise rather than silently passing.
    assert_eq!(
        module_handle("winmm.dll"),
        0,
        "winmm.dll must be unmapped before install"
    );

    // SAFETY: `handle` is a live module handle from `LoadLibraryW`.
    let module = unsafe { Module::from_handle(handle.cast()) }.expect("acquire delay fixture");

    // The engine must enumerate the winmm!timeGetTime import as delay-load.
    let imports = module.imports().expect("enumerate imports");
    let has_delay_slot = imports.iter().any(|s| {
        s.kind() == ImportKind::DelayLoad
            && s.library().eq_ignore_ascii_case("winmm.dll")
            && s.symbol() == Some(&Symbol::name("timeGetTime"))
    });
    assert!(
        has_delay_slot,
        "engine must enumerate the winmm!timeGetTime delay import"
    );

    let call_delayed = export(handle, "call_timegettime");

    // Install a required hook on the delay slot while winmm is still unmapped.
    let guard = install(
        &module,
        &[Replacement::by_name(
            "winmm.dll",
            "timeGetTime",
            fake_time as *const c_void,
        )],
    )
    .expect("install delay hook");

    // Installing force-loaded winmm to resolve the real original authoritatively.
    let winmm_base = module_handle("winmm.dll");
    assert_ne!(
        winmm_base, 0,
        "install force-loaded winmm.dll to resolve the original"
    );

    // Calls through the fixture now route to our replacement.
    assert_eq!(
        call(call_delayed),
        0xDEAD_BEEF,
        "delay slot routed to the replacement"
    );

    // `original()` must be the REAL winmm export, never the delay stub: a stub
    // lives in the fixture module, the real export lives in winmm.dll.
    let original = guard
        .original(&Symbol::name("timeGetTime"))
        .expect("original resolved");
    assert!(!original.is_null(), "original must be non-null");
    // SAFETY: `original` points into a currently-loaded module (winmm).
    let original_module = unsafe { Module::from_address(original) }.expect("original in a module");
    assert_eq!(
        original_module.base() as usize,
        winmm_base,
        "original points into winmm.dll, not the delay stub in the fixture",
    );
    assert!(original_module.name().eq_ignore_ascii_case("winmm.dll"));

    // Passing through `original()` must NOT clobber our hook. If `original()` were
    // the `__delayLoadHelper2` stub, this call would resolve winmm!timeGetTime and
    // write it back into THIS slot, silently uninstalling the hook.
    // SAFETY: `original` is winmm!timeGetTime, an `extern "system" fn() -> u32`.
    let _real_ticks = unsafe { core::mem::transmute::<_, ZeroArgU32>(original)() };
    assert_eq!(
        call(call_delayed),
        0xDEAD_BEEF,
        "hook survived passthrough (original() was the real callee, not a self-clobbering stub)",
    );

    // Uninstall restores the delay slot; the fixture reaches the real winmm again.
    guard.uninstall().expect("uninstall delay hook");
    assert_ne!(
        call(call_delayed),
        0xDEAD_BEEF,
        "delay slot restored to the real callee"
    );
}
