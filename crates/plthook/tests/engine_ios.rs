//! iOS/iPadOS Mach-O engine integration test.
//!
//! The test hooks this binary's `getenv` import, verifies the replacement is called,
//! and confirms that uninstalling restores the original import. It is compiled and
//! run when the test target is iOS; parser coverage remains available on every host.

#![cfg(target_os = "ios")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::ffi::{c_char, c_void};
use plthook::{Module, Replacement, Symbol, install};

/// A sentinel the replacement returns, so a redirect is unmistakable.
static FAKE_ENV: &[u8] = b"HOOKFS_FAKE_GETENV\0";

/// The replacement `getenv`: ignores its argument and returns the sentinel.
extern "C" fn fake_getenv(_name: *const c_char) -> *mut c_char {
    FAKE_ENV.as_ptr().cast::<c_char>().cast_mut()
}

/// An address inside this test binary, used to acquire its image.
extern "C" fn anchor() {}

#[test]
fn hook_getenv_in_this_binary() {
    // Reference `getenv` from this image so its import slot definitely exists, and
    let probe = c"PATH";
    // SAFETY: `probe` is a valid NUL-terminated C string.
    let real = unsafe { libc::getenv(probe.as_ptr()) };

    // Acquire this executable's Mach-O image by an address inside it (via `dladdr`).
    // SAFETY: `anchor` is a live function in this mapped image.
    let module = unsafe { Module::from_address(anchor as *const c_void) }
        .expect("acquire the test binary Mach-O image");

    // Enumerate imports; `getenv` must be present as a rebindable symbol pointer.
    // A missing slot is a real engine regression, so this fails rather than skips.
    let imports = module.imports().expect("enumerate Mach-O imports");
    let getenv = imports
        .iter()
        .find(|slot| slot.symbol() == Some(&Symbol::name("getenv")))
        .expect(
            "getenv must be a Mach-O import of this test binary A?€�t the live engine 
             cannot be exercised without it",
        );
    // On plain arm64 (the standard iOS device target) the slot is not authenticated;
    // an arm64e slice would flag it and `install` would refuse it (R1).
    assert!(
        !getenv.is_authenticated(),
        "a plain arm64 import must not be flagged authenticated (arm64e is refused, R1)"
    );

    // Install the hook transactionally: this flips the slot page's protection via
    // `mach_vm_protect` (VM_PROT_COPY for a const `__DATA_CONST`/`__got` page A?€�t
    // DATA only, never executable, so no W^X/code-signing rule is touched), then
    // atomically swaps the pointer.
    let guard = install(
        &module,
        &[Replacement::by_symbol(
            "getenv",
            fake_getenv as *const c_void,
        )],
    )
    .expect("install getenv hook");

    // The redirect is observable: this process's own `getenv()` now returns the
    // sentinel (routed through the rebound symbol-pointer slot).
    // SAFETY: `probe` is valid.
    let hooked = unsafe { libc::getenv(probe.as_ptr()) };
    assert_eq!(
        hooked.cast_const().cast::<u8>(),
        FAKE_ENV.as_ptr(),
        "getenv should be redirected to the replacement"
    );

    // The saved original is the real libc `getenv` (resolved via `dlsym`/the bound
    // slot, not a stub); calling it returns the true value captured before hooking.
    let original = guard
        .original(&Symbol::name("getenv"))
        .expect("original recorded");
    // SAFETY: `original` is the real `getenv` entry point.
    let real_getenv: extern "C" fn(*const c_char) -> *mut c_char =
        unsafe { std::mem::transmute(original) };
    assert_eq!(
        real_getenv(probe.as_ptr()),
        real,
        "the saved original returns the true value"
    );

    // Restore (compare-exchange) and confirm the real behavior returns.
    guard.uninstall().expect("uninstall restores the slot");
    // SAFETY: `probe` is valid.
    assert_eq!(
        unsafe { libc::getenv(probe.as_ptr()) },
        real,
        "getenv restored to the real value"
    );
}
