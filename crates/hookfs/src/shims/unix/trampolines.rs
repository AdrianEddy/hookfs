//! Per-arch `global_asm!` tail-call trampolines for the C-variadic `open` family
//!
//! `open`/`open64`/`openat` are declared `int open(const char *, int, ...)`: the
//! optional `mode` argument is present only when `O_CREAT`/`O_TMPFILE` is set.
//! Stable Rust cannot *define* a C-variadic function, so the installed symbol is a
//! tiny hand-written trampoline that tail-jumps into a fixed-arity Rust entry point.
//! That entry reads the (possibly-unused) `mode` from its third/fourth intege
//! parameter A?€�t the register `x2`/`rdx` (`open`) or `x3`/`rcx` (`openat`) where a
//! fixed-arity C function receives it. Where the calling convention already places
//! the first variadic integer argument in *that same register*, the trampoline is a
//! bare tail-jump preserving every register; where it does not, the trampoline first
//! loads `mode` into that register from the stack:
//!
//! * **`SysV` `AMD64`** (Linux & macOS x86-64): integer args A?€�t fixed *and*
//!   variadic A?€�t in `rdi, rsi, rdx, rcx, A?€¦`; `open(path, flags, mode)` A?�?’ `mode`
//!   already in `rdx`. A bare `jmp` preserves all of them.
//! * **`AAPCS64`** (Linux arm64): integer args in `x0, x1, x2, A?€¦`; variadic intege
//!   args are *not* special-cased, so `open(path, flags, mode)` A?�?’ `mode` already in
//!   `x2`. A bare `b` preserves them.
//! * **Apple arm64** (macOS **and** iOS/iPadOS device) diverges from `AAPCS64`:
//!   *variadic* arguments are passed on the **stack**, never in the argument registers
//!   (per Apple's "Writing ARM64 Code for Apple Platforms"; this is why `va_list` is a
//!   plain `char*` there). The fixed args still occupy the registers, so `open`'s
//!   `mode` is the first stack slot at `[sp]` (and `openat`'s is likewise the first
//!   slot). The trampoline `ldr`s it into `x2`/`x3` before the `b` A?€�t see the
//!   Apple+aarch64 block below. This is the trampoline that carries on `aarch64-apple-
//!   ios` (the primary iPadOS device target).
//!
//! # Platform differences
//! The two Unix object formats name and hide symbols differently, so the asm is
//! cfg-split: ELF (Linux) uses bare labels + `.hidden`; Mach-O (macOS **and** iOS)
//! uses `_`-prefixed labels + `.private_extern` (the compiler maps the `extern "C"`
//! names to the underscored symbols) A?€�t the identical Apple asm serves both Darwin
//! targets. Darwin has no `open64`, so that trampoline is Linux-only.
//!
//! When the caller passed only two arguments (no `O_CREAT`), `mode` is
//! indeterminate A?€�t a stale register on `AMD64`/`AAPCS64`, or an unrelated in-bounds
//! stack word on Apple arm64 (`[sp]` is always mapped, so the `ldr` never faults).
//! It is never read on the virtual (read-only) path, and forwarded verbatim to the
//! *original* variadic `open` on passthrough, where the kernel ignores it because
//! `O_CREAT` is absent.

use super::{PfnOpen, PfnOpenat};
use crate::dispatch::{HookScope, Sym, guard_abi};
use crate::namespace::decode_cstr;
use crate::router::Route;
use core::ffi::{c_char, c_int, c_void};
use libc::mode_t;

/// The `errno` used when a trampoline entry's Rust work panics.
const PANIC_ERRNO: u32 = libc::EIO as u32;

unsafe extern "C" {
    /// The installed replacement for `open` (tail-jumps to [`open_entry`]).
    fn hookfs_open_trampoline();
    /// The installed replacement for `openat` (tail-jumps to [`openat_entry`]).
    fn hookfs_openat_trampoline();
    /// The installed replacement for `open64` (tail-jumps to [`open64_entry`]) A?€�t
    /// Linux only.
    #[cfg(target_os = "linux")]
    fn hookfs_open64_trampoline();
}

// ---- ELF (Linux): bare labels + `.hidden`; includes open64 ------------------

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
core::arch::global_asm!(
    ".p2align 4",
    ".globl hookfs_open_trampoline",
    ".hidden hookfs_open_trampoline",
    "hookfs_open_trampoline:",
    "    jmp {open}",
    ".globl hookfs_open64_trampoline",
    ".hidden hookfs_open64_trampoline",
    "hookfs_open64_trampoline:",
    "    jmp {open64}",
    ".globl hookfs_openat_trampoline",
    ".hidden hookfs_openat_trampoline",
    "hookfs_openat_trampoline:",
    "    jmp {openat}",
    open = sym open_entry,
    open64 = sym open64_entry,
    openat = sym openat_entry,
);

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
core::arch::global_asm!(
    ".p2align 4",
    ".globl hookfs_open_trampoline",
    ".hidden hookfs_open_trampoline",
    "hookfs_open_trampoline:",
    "    b {open}",
    ".globl hookfs_open64_trampoline",
    ".hidden hookfs_open64_trampoline",
    "hookfs_open64_trampoline:",
    "    b {open64}",
    ".globl hookfs_openat_trampoline",
    ".hidden hookfs_openat_trampoline",
    "hookfs_openat_trampoline:",
    "    b {openat}",
    open = sym open_entry,
    open64 = sym open64_entry,
    openat = sym openat_entry,
);

// ---- Mach-O (macOS + iOS): `_`-prefixed labels + `.private_extern`; no open64 --

#[cfg(all(any(target_os = "macos", target_os = "ios"), target_arch = "x86_64"))]
core::arch::global_asm!(
    ".p2align 4",
    ".globl _hookfs_open_trampoline",
    ".private_extern _hookfs_open_trampoline",
    "_hookfs_open_trampoline:",
    "    jmp {open}",
    ".globl _hookfs_openat_trampoline",
    ".private_extern _hookfs_openat_trampoline",
    "_hookfs_openat_trampoline:",
    "    jmp {openat}",
    open = sym open_entry,
    openat = sym openat_entry,
);

#[cfg(all(any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
core::arch::global_asm!(
    ".p2align 4",
    ".globl _hookfs_open_trampoline",
    ".private_extern _hookfs_open_trampoline",
    "_hookfs_open_trampoline:",
    // Apple's arm64 ABI passes *variadic* arguments on the stack, never in the
    // argument registers (unlike standard AAPCS64). `open`'s two fixed args occupy
    // x0/x1; its first A?€�t and only A?€�t variadic arg, `mode`, is the first stack slot,
    // at `[sp]`. Load it into `w2`, the register `open_entry`'s fixed `mode`
    // parameter reads. `ldr w` reads exactly the promoted 4-byte `int` and
    // zero-extends into `x2`, matching the register content the Linux path already
    // carries; `sp` is untouched, so the `b` tail-jump preserves stack alignment.
    "    ldr w2, [sp]",
    "    b {open}",
    ".globl _hookfs_openat_trampoline",
    ".private_extern _hookfs_openat_trampoline",
    "_hookfs_openat_trampoline:",
    // `openat`'s three fixed args occupy x0..x2; its first variadic arg, `mode`, is
    // the first stack slot at `[sp]`. Load it into `w3` for `openat_entry`.
    "    ldr w3, [sp]",
    "    b {openat}",
    open = sym open_entry,
    openat = sym openat_entry,
);

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("the variadic-open trampoline supports only x86_64 and aarch64");

/// The address to install for `sym` (an `open`-family trampoline). Only the open
/// family routes here; any other symbol falls back to the `open` trampoline.
pub(super) fn trampoline_address(sym: Sym) -> *const c_void {
    let f: unsafe extern "C" fn() = match sym {
        #[cfg(target_os = "linux")]
        Sym::Open64 => hookfs_open64_trampoline,
        Sym::Openat => hookfs_openat_trampoline,
        _ => hookfs_open_trampoline, // Sym::Open (and, unreachable, anything else)
    };
    f as *const c_void
}

/// Fixed-arity Rust entry for `open`, tail-called by the trampoline.
extern "C" fn open_entry(path: *const c_char, flags: c_int, mode: mode_t) -> c_int {
    open_impl(Sym::Open, path, flags, mode)
}

/// Fixed-arity Rust entry for `open64` (Linux), tail-called by the trampoline.
#[cfg(target_os = "linux")]
extern "C" fn open64_entry(path: *const c_char, flags: c_int, mode: mode_t) -> c_int {
    open_impl(Sym::Open64, path, flags, mode)
}

/// Shared `open`/`open64` logic. `sym` selects which original to forward to.
fn open_impl(sym: Sym, path: *const c_char, flags: c_int, mode: mode_t) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnOpen = super::orig_fn(sym);
            // SAFETY: re-issue the exact variadic call to the real `open`; `mode` is
            // forwarded verbatim (ignored by the kernel unless `O_CREAT` is set).
            // Promote `mode_t` to `c_uint` for the variadic slot A?€�t the C default
            // argument promotion, and required on macOS where `mode_t` is `u16`.
            unsafe { orig(path, flags, libc::c_uint::from(mode)) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: `path` is the caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match super::route(&decoded) {
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                super::set_errno(libc::ENOENT);
                -1
            }
            Some((engine, Route::Virtual(p))) => super::open_virtual_fd(&engine, &p, flags),
        }
    })
}

/// Fixed-arity Rust entry for `openat`, tail-called by the trampoline.
extern "C" fn openat_entry(dirfd: c_int, path: *const c_char, flags: c_int, mode: mode_t) -> c_int {
    guard_abi(-1, PANIC_ERRNO, || {
        let pass = || {
            let orig: PfnOpenat = super::orig_fn(Sym::Openat);
            // SAFETY: re-issue the exact variadic call to the real `openat`. `mode`
            // is promoted to `c_uint` (C default promotion; `mode_t` is `u16` on
            // macOS, which cannot occupy a variadic slot directly).
            unsafe { orig(dirfd, path, flags, libc::c_uint::from(mode)) }
        };
        let Some(_scope) = HookScope::enter() else {
            return pass();
        };
        // SAFETY: caller's NUL-terminated path (or null).
        let Some(decoded) = (unsafe { decode_cstr(path) }) else {
            return pass();
        };
        match super::route(&decoded) {
            // A relative path routes Real (no virtual cwd) and passes through with
            // the caller's `dirfd` honored; an absolute virtual path ignores `dirfd`.
            None | Some((_, Route::Real)) => pass(),
            Some((_, Route::Rejected)) => {
                super::set_errno(libc::ENOENT);
                -1
            }
            Some((engine, Route::Virtual(p))) => super::open_virtual_fd(&engine, &p, flags),
        }
    })
}
