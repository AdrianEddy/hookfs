//! the test binary's own module, enumerate its libc imports (with symbol
//! versions), then transactionally rebind one and prove the redirect, the
//! `/proc/self/maps` + `mprotect` RELRO handling, and the compare-exchange restore.
//!
//! This runs on **native Linux** (CI and WSL); on the Windows development host it
//! is cfg-compiled only. It targets **`getenv`** — a plain libc function the Rust
//! toolchain resolves through this executable's own GOT (a `GLOB_DAT`/`JUMP_SLOT`
//! import): unlike the syscall-y candidates, `getenv` is never inlined by the
//! compiler, never satisfied by the vDSO, and is not an IFUNC, so it is reliably a
//! real dynamic import of the test binary. Hooking that slot redirects this
//! process's own `getenv()` — a self-contained end-to-end exercise of the live ELF
//! path (`dl_iterate_phdr` acquisition, live RELA/version parsing including
//! `DT_VERNEED`, `dlsym(RTLD_DEFAULT)` originals, aligned atomic swap, restore).
//!
//! The test **fails** (never silently skips) if the engine cannot find, hook, or
//! restore the slot: that is precisely the live-engine regression this test exists
//! to catch.

#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::ffi::{c_char, c_void};
use plthook::{Module, Replacement, Symbol, install};

/// A sentinel value the replacement returns, so a redirect is unmistakable.
static FAKE_ENV: &[u8] = b"HOOKFS_FAKE_GETENV\0";

/// The replacement `getenv`: ignores its argument and returns the sentinel.
extern "C" fn fake_getenv(_name: *const c_char) -> *mut c_char {
    FAKE_ENV.as_ptr().cast::<c_char>().cast_mut()
}

/// An address inside this test binary, used to acquire its module.
extern "C" fn anchor() {}

#[test]
fn hook_getenv_in_this_binary() {
    // Reference `getenv` from this image so its GOT slot definitely exists, and
    // comparison. `PATH` is set in every sane environment, but the assertions
    // compare pointers and so hold even if it were unset (both `null`).
    let probe = c"PATH";
    let real = unsafe { libc::getenv(probe.as_ptr()) };

    // Acquire this executable's module by an address inside it.
    let module = unsafe { Module::from_address(anchor as *const c_void) }
        .expect("acquire the test binary module");

    // Enumerate imports; `getenv` must be present (a guaranteed dynamic import),
    // carrying a GLIBC_* version. A missing slot is a real engine regression, so
    // this fails rather than skips.
    let imports = module.imports().expect("enumerate ELF imports");
    let getenv = imports
        .iter()
        .find(|slot| slot.symbol() == Some(&Symbol::name("getenv")))
        .expect(
            "getenv must be a GOT/PLT import of this test binary — the live ELF \
             engine cannot be exercised without it",
        );
    assert!(
        getenv.version().is_some_and(|v| v.starts_with("GLIBC")),
        "expected a GLIBC symbol version, got {:?}",
        getenv.version()
    );

    // Install the hook transactionally (this flips the GOT page's protection via
    // `mprotect`, handling RELRO, then atomically swaps the slot).
    let guard = install(
        &module,
        &[Replacement::by_symbol(
            "getenv",
            fake_getenv as *const c_void,
        )],
    )
    .expect("install getenv hook");

    // The redirect is observable: this process's own `getenv()` now returns the
    // sentinel (routed through the rebound GOT slot).
    let hooked = unsafe { libc::getenv(probe.as_ptr()) };
    assert_eq!(
        hooked.cast_const().cast::<u8>(),
        FAKE_ENV.as_ptr(),
        "getenv should be redirected to the replacement"
    );

    // The saved original is the real libc `getenv` (resolved via dlsym, not the
    // slot), and calling it returns the true value captured before hooking.
    let original = guard
        .original(&Symbol::name("getenv"))
        .expect("original recorded");
    let real_getenv: extern "C" fn(*const c_char) -> *mut c_char =
        unsafe { std::mem::transmute(original) };
    assert_eq!(
        real_getenv(probe.as_ptr()),
        real,
        "the saved original returns the true value"
    );

    // Restore (compare-exchange) and confirm the real behavior returns.
    guard.uninstall().expect("uninstall restores the slot");
    assert_eq!(
        unsafe { libc::getenv(probe.as_ptr()) },
        real,
        "getenv restored to the real value"
    );
}
