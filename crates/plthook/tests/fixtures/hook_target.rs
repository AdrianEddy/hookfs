//! Engine test fixture — compiled to a `cdylib` by `build.rs`.
//!
//! Each exported function calls a KERNEL32 import **through the module's IAT**
//! (an external DLL call cannot be inlined to a direct address), so the engine
//! can rebind that slot and a test can observe the redirect, then restore it.
//!
//! Built with edition 2021 (plain `extern` blocks / `#[no_mangle]`) and a static
//! CRT so the DLL is self-contained.
#![crate_type = "cdylib"]

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcessId() -> u32;
    fn GetCurrentThreadId() -> u32;
}

/// Returns the current process id via `KERNEL32!GetCurrentProcessId`.
#[no_mangle]
#[inline(never)]
pub extern "system" fn fixture_pid() -> u32 {
    unsafe { GetCurrentProcessId() }
}

/// Returns the current thread id via `KERNEL32!GetCurrentThreadId`.
#[no_mangle]
#[inline(never)]
pub extern "system" fn fixture_tid() -> u32 {
    unsafe { GetCurrentThreadId() }
}
