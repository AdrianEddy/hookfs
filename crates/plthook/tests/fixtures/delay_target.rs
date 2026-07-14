//! Delay-load engine test fixture — compiled to a `cdylib` by `build.rs` with
//! `/DELAYLOAD:winmm.dll` (+ `delayimp.lib`).
//!
//! `timeGetTime` is imported from `winmm.dll` through the PE **delay-load**
//! directory (`IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT`). Its delay IAT slot holds
//! the `__delayLoadHelper2` load thunk until the first call, at which point the
//! helper would `LoadLibrary("winmm")`, `GetProcAddress`, patch the slot, and
//! tail-call. Loading this DLL therefore does **not** load `winmm.dll`: it stays
//! unloaded until `call_timegettime` is first called — or until the engine
//! force-loads it to resolve the original. That is exactly the shape the engine's
//! delay-original resolution must get right, since the pre-write slot value is a
//! stub, never the real callee.
//!
//! Built with edition 2021 (plain `extern` blocks / `#[no_mangle]`) and a static
//! CRT so the DLL is self-contained.
#![crate_type = "cdylib"]

#[link(name = "winmm")]
extern "system" {
    fn timeGetTime() -> u32;
}

/// Calls `winmm!timeGetTime` **through the module's delay-load IAT slot**, so the
/// engine can rebind that slot and a test can observe the redirect, then restore.
#[no_mangle]
#[inline(never)]
pub extern "system" fn call_timegettime() -> u32 {
    unsafe { timeGetTime() }
}
