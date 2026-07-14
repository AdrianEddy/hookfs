//! End-to-end engine tests against real loaded fixture DLLs (built by `build.rs`).
//!
//! These exercise the whole install → observe-redirect → restore lifecycle on a
//! genuine module IAT: single/multiple symbols, guard drop, nested LIFO restore,
//! the compare-exchange conflict path, two modules (two slots for one symbol),
//! and a concurrency stress (calls through a slot while it is repeatedly
//! installed/uninstalled). Everything runs in one sequential test because all
//! scenarios mutate one shared process-global resource — the fixtures' import
//! tables — so serializing them keeps the assertions deterministic.
#![cfg(windows)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::missing_transmute_annotations,
    clippy::too_many_lines
)]

use core::ffi::c_void;
use plthook::{Error, Module, Replacement, Symbol, install};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

/// A `fixture_pid` / `fixture_tid`-shaped export.
type ZeroArgU32 = unsafe extern "system" fn() -> u32;

// Distinct, recognizable replacement return values.
extern "system" fn fake_pid() -> u32 {
    0xDEAD_BEEF
}
extern "system" fn fake_pid2() -> u32 {
    0xFEED_FACE
}
extern "system" fn fake_tid() -> u32 {
    0x1234_5678
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn load(path: &str) -> HMODULE {
    let wide = wide(path);
    // SAFETY: `wide` is a valid NUL-terminated path.
    let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
    assert!(!handle.is_null(), "LoadLibraryW failed for {path}");
    handle
}

/// Resolve an export and transmute it to a callable `extern "system" fn() -> u32`.
fn export(handle: HMODULE, name: &str) -> ZeroArgU32 {
    let mut c_name = name.as_bytes().to_vec();
    c_name.push(0);
    // SAFETY: `c_name` is a valid NUL-terminated symbol name; `handle` is live.
    let proc = unsafe { GetProcAddress(handle, c_name.as_ptr()) };
    let proc = proc.unwrap_or_else(|| panic!("export `{name}` not found"));
    // SAFETY: the fixture declares these exports as `extern "system" fn() -> u32`.
    unsafe { core::mem::transmute::<_, ZeroArgU32>(proc) }
}

fn call(f: ZeroArgU32) -> u32 {
    // SAFETY: `f` is a valid zero-argument `extern "system"` function.
    unsafe { f() }
}

fn pid_replacement(f: extern "system" fn() -> u32) -> Replacement {
    Replacement::by_name("KERNEL32.dll", "GetCurrentProcessId", f as *const c_void)
}

/// Current page protection of the `KERNEL32!GetCurrentProcessId` import slot,
/// used to prove the engine restores the original protection exactly.
fn pid_slot_protection(module: &Module) -> u32 {
    module
        .imports()
        .expect("enumerate")
        .iter()
        .find(|s| {
            s.symbol() == Some(&Symbol::name("GetCurrentProcessId"))
                && s.library().eq_ignore_ascii_case("KERNEL32.dll")
        })
        .expect("pid slot")
        .protection()
}

/// The module-acquisition paths (`from_handle`, `from_address`, `from_name`,
/// `enumerate`) agree and the enumerator sees the process's loaded modules.
/// Read-only, so it is safe to run alongside the patching lifecycle test.
#[test]
fn module_acquisition_paths_agree() {
    let handle = load(env!("PLTHOOK_FIXTURE_DLL"));
    // SAFETY: `handle` is a live module handle.
    let by_handle = unsafe { Module::from_handle(handle.cast()) }.expect("from_handle");

    // An address inside the fixture resolves to the same module base.
    let pid_export = export(handle, "fixture_pid") as *const c_void;
    // SAFETY: `pid_export` points into the fixture's code.
    let by_address = unsafe { Module::from_address(pid_export) }.expect("from_address");
    assert_eq!(by_handle.base(), by_address.base(), "handle/address agree");

    // KERNEL32 is always loaded; `from_name` and `enumerate` must both find it.
    let kernel32 = Module::from_name("kernel32.dll").expect("from_name kernel32");
    assert!(kernel32.name().eq_ignore_ascii_case("kernel32.dll"));

    let modules = Module::enumerate().expect("enumerate modules");
    assert!(modules.len() > 1, "process has several modules loaded");
    assert!(
        modules.iter().any(|m| m.base() == kernel32.base()),
        "enumeration includes KERNEL32",
    );
    assert!(
        modules.iter().any(|m| m.base() == by_handle.base()),
        "enumeration includes the fixture",
    );
}

#[test]
fn engine_lifecycle() {
    let path_a = env!("PLTHOOK_FIXTURE_DLL");
    let path_b = env!("PLTHOOK_FIXTURE_DLL_B");

    let handle_a = load(path_a);
    let handle_b = load(path_b);
    // SAFETY: `handle_a`/`handle_b` are live module handles from `LoadLibraryW`.
    let module_a = unsafe { Module::from_handle(handle_a.cast()) }.expect("acquire fixture A");
    let module_b = unsafe { Module::from_handle(handle_b.cast()) }.expect("acquire fixture B");

    let pid_a = export(handle_a, "fixture_pid");
    let tid_a = export(handle_a, "fixture_tid");
    let pid_b = export(handle_b, "fixture_pid");

    let real_pid = std::process::id();
    let real_tid = call(tid_a);

    // module identity sanity: name matches the built fixture.
    assert!(
        module_a
            .name()
            .eq_ignore_ascii_case("plthook_hook_target.dll")
    );

    // --- Scenario 1: single-symbol redirect, then restore ------------------
    // Baseline goes through the real KERNEL32 import.
    assert_eq!(
        call(pid_a),
        real_pid,
        "baseline must be the real process id"
    );
    let protection_before = pid_slot_protection(&module_a);
    {
        let guard = install(&module_a, &[pid_replacement(fake_pid)]).expect("install pid hook");
        assert_eq!(
            call(pid_a),
            0xDEAD_BEEF,
            "call must route to the replacement"
        );
        // The engine hands the caller the canonical original for pass-through.
        let original = guard
            .original(&Symbol::name("GetCurrentProcessId"))
            .expect("original resolved");
        // SAFETY: `original` is `GetCurrentProcessId`, an `extern "system" fn() -> u32`.
        let original_pid = unsafe { core::mem::transmute::<_, ZeroArgU32>(original)() };
        assert_eq!(
            original_pid, real_pid,
            "original entry point still yields the real pid"
        );
        guard.uninstall().expect("uninstall pid hook");
    }
    assert_eq!(call(pid_a), real_pid, "slot restored after uninstall");
    // IAT sits on a real read-only page, so this is a genuine RO→RW→RO cycle.
    assert_eq!(
        pid_slot_protection(&module_a),
        protection_before,
        "page protection restored exactly after an install/uninstall cycle",
    );

    // --- Scenario 2: dropping the guard restores ---------------------------
    {
        let _guard = install(&module_a, &[pid_replacement(fake_pid)]).expect("install");
        assert_eq!(call(pid_a), 0xDEAD_BEEF);
        // guard dropped here
    }
    assert_eq!(call(pid_a), real_pid, "Drop restored the slot");

    // --- Scenario 3: multiple symbols in one module ------------------------
    {
        let guard = install(
            &module_a,
            &[
                pid_replacement(fake_pid),
                Replacement::by_name(
                    "KERNEL32.dll",
                    "GetCurrentThreadId",
                    fake_tid as *const c_void,
                ),
            ],
        )
        .expect("install two hooks");
        assert_eq!(call(pid_a), 0xDEAD_BEEF);
        assert_eq!(call(tid_a), 0x1234_5678);
        assert_eq!(guard.installed().len(), 2);
        guard.uninstall().expect("uninstall two hooks");
    }
    assert_eq!(call(pid_a), real_pid);
    assert_eq!(call(tid_a), real_tid);

    // --- Scenario 4: nested guards, LIFO uninstall fully restores ----------
    {
        let outer = install(&module_a, &[pid_replacement(fake_pid)]).expect("install A");
        assert_eq!(call(pid_a), 0xDEAD_BEEF);
        let inner = install(&module_a, &[pid_replacement(fake_pid2)]).expect("install B");
        assert_eq!(call(pid_a), 0xFEED_FACE);
        inner.uninstall().expect("uninstall inner");
        assert_eq!(
            call(pid_a),
            0xDEAD_BEEF,
            "inner restored to its prior (outer's replacement)"
        );
        outer.uninstall().expect("uninstall outer");
        assert_eq!(call(pid_a), real_pid, "outer restored to the true original");
    }

    // --- Scenario 5: two modules => two slots for the same symbol ----------
    {
        let g_a = install(&module_a, &[pid_replacement(fake_pid)]).expect("install A");
        let g_b = install(&module_b, &[pid_replacement(fake_pid2)]).expect("install B");
        assert_eq!(call(pid_a), 0xDEAD_BEEF, "module A slot independent");
        assert_eq!(call(pid_b), 0xFEED_FACE, "module B slot independent");
        g_b.uninstall().expect("uninstall B");
        g_a.uninstall().expect("uninstall A");
        assert_eq!(call(pid_a), real_pid);
        assert_eq!(call(pid_b), real_pid);
    }

    // --- Scenario 6: non-LIFO restore reports a conflict (no clobber) ------
    // Run on module B so the intentionally-stranded slot never affects module A.
    {
        let first = install(&module_b, &[pid_replacement(fake_pid)]).expect("install first");
        let second = install(&module_b, &[pid_replacement(fake_pid2)]).expect("install second");
        assert_eq!(call(pid_b), 0xFEED_FACE);
        // Restoring `first` out of order must NOT clobber `second`.
        let conflict = first.uninstall();
        assert!(
            matches!(conflict, Err(Error::RestoreConflict { .. })),
            "out-of-order restore must report a conflict, got {conflict:?}",
        );
        assert_eq!(
            call(pid_b),
            0xFEED_FACE,
            "the subsequent hook was preserved"
        );
        // `second` restores to its own prior (first's replacement); B is left
        second.uninstall().expect("uninstall second");
        assert_eq!(call(pid_b), 0xDEAD_BEEF);
    }

    // --- Scenario 7: concurrency stress (R20) ------------------------------
    // Threads hammer the slot while the main thread installs/uninstalls; a
    // reader must only ever observe the real or the replacement pointer, never a
    // torn/garbage value (aligned atomic swap + page-batched protection).
    {
        let stop = Arc::new(AtomicBool::new(false));
        let bad = Arc::new(AtomicU32::new(0));
        let saw_bad = Arc::new(AtomicBool::new(false));
        let pid_addr = pid_a as usize;

        let mut workers = Vec::new();
        for _ in 0..4 {
            let stop = Arc::clone(&stop);
            let bad = Arc::clone(&bad);
            let saw_bad = Arc::clone(&saw_bad);
            workers.push(std::thread::spawn(move || {
                // SAFETY: `pid_addr` is the fixture's `extern "system" fn() -> u32`.
                let f: ZeroArgU32 = unsafe { core::mem::transmute(pid_addr) };
                while !stop.load(Ordering::Relaxed) {
                    let value = unsafe { f() };
                    if value != real_pid && value != 0xDEAD_BEEF {
                        bad.store(value, Ordering::Relaxed);
                        saw_bad.store(true, Ordering::Relaxed);
                    }
                }
            }));
        }

        for _ in 0..300 {
            let guard = install(&module_a, &[pid_replacement(fake_pid)]).expect("stress install");
            guard.uninstall().expect("stress uninstall");
        }
        stop.store(true, Ordering::Relaxed);
        for worker in workers {
            worker.join().expect("worker joined");
        }

        assert!(
            !saw_bad.load(Ordering::Relaxed),
            "a call observed a torn/garbage pointer: {:#x}",
            bad.load(Ordering::Relaxed)
        );
        assert_eq!(call(pid_a), real_pid, "slot restored after stress");
    }
}
