//! Import slots, replacements, the installed-hook guard, and the transactional
//!
//! This module is platform-agnostic apart from the two protection primitives it
//! calls in [`crate::sys`] (`make_writable` / `restore_protection`) and the
//! original-symbol resolver. The transaction — resolve-all-first, page-batched
//! protection changes, atomic aligned swap, rollback on failure, and
//! compare-exchange restore — is exactly what the ELF and Mach-O backends will
//! reuse, swapping only the underlying `sys` calls.
//!
//! ## Concurrency model
//! A single process-wide [`install_lock`] serializes every install, uninstall,
//! and protection change, so slot *writes* never race each other. Calls *through*
//! a slot are lock-free: the target's compiled code reads the slot as an aligned
//! pointer-width load, and every write we make is a single aligned atomic store
//! (via [`AtomicUsize`]). Aligned pointer-width loads/stores are atomic on all
//! supported ISAs (x86-64, aarch64), so a concurrent call observes either the
//! old or the new pointer — never a torn value. No allocation, logging, or lock
//! acquisition happens on the call-through path (there is none: we only patch
//! data).

use crate::error::{Error, Result};
use crate::import::RawImport;
use crate::module::Module;
use crate::{ImportKind, Symbol, sys};
use core::ffi::c_void;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// The process-wide install/uninstall/protection lock. A poisoned lock is
/// recovered (we never leave engine state inconsistent under it), so a panic in
/// unrelated code cannot wedge the engine.
fn install_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Atomically swap the pointer stored in `slot`, returning the previous value.
///
/// # Safety
/// `slot` must be a pointer-aligned, writable, in-bounds import slot (its page
/// made writable by [`PageProtector`]). The engine holds [`install_lock`], so
/// this is the only writer.
unsafe fn slot_swap(slot: usize, new: usize) -> usize {
    // SAFETY: caller guarantees `slot` is a valid, aligned, writable pointer cell.
    let atomic = unsafe { &*(slot as *const AtomicUsize) };
    atomic.swap(new, Ordering::AcqRel)
}

/// Compare-exchange restore: store `new` only if the slot still holds `expected`.
///
/// # Safety
/// Same requirements as [`slot_swap`].
unsafe fn slot_compare_exchange(
    slot: usize,
    expected: usize,
    new: usize,
) -> core::result::Result<usize, usize> {
    // SAFETY: caller guarantees `slot` is a valid, aligned, writable pointer cell.
    let atomic = unsafe { &*(slot as *const AtomicUsize) };
    atomic.compare_exchange(expected, new, Ordering::AcqRel, Ordering::Acquire)
}

/// Read the current pointer in a slot (the pre-write "original" fallback).
///
/// # Safety
/// `slot` must be a readable, pointer-aligned import slot.
unsafe fn slot_load(slot: usize) -> usize {
    // SAFETY: caller guarantees `slot` is a valid, aligned, readable pointer cell.
    let atomic = unsafe { &*(slot as *const AtomicUsize) };
    atomic.load(Ordering::Acquire)
}

/// A batch of page-protection changes, captured so they can be restored exactly.
///
/// Protections are captured and restored **per unique page**, not per slot:
/// several slots typically share one page, and capturing each page's original
/// protection exactly once is the only way to restore it correctly regardless of
struct PageProtector {
    /// `(page_start, original_protection)`, in the order pages were made writable.
    pages: Vec<(usize, u32)>,
}

impl PageProtector {
    /// Make every page spanned by `slots` writable, capturing prior protections.
    /// On any failure, restore the pages already changed and return the error —
    /// so a failed call leaves no page writable.
    fn make_writable(slots: &[usize]) -> Result<Self> {
        let page_size = sys::page_size();
        let mask = !(page_size - 1);
        let mut pages: Vec<usize> = Vec::new();
        for &slot in slots {
            let first = slot & mask;
            let last = slot.saturating_add(crate::arch::PTR_SIZE - 1) & mask;
            let mut page = first;
            loop {
                if !pages.contains(&page) {
                    pages.push(page);
                }
                if page == last {
                    break;
                }
                page = page.saturating_add(page_size);
            }
        }

        let mut done: Vec<(usize, u32)> = Vec::with_capacity(pages.len());
        for &page in &pages {
            // SAFETY: `page` is a page-aligned address derived from a validated,
            // committed import slot.
            match unsafe { sys::make_writable(page) } {
                Ok(old) => done.push((page, old)),
                Err(err) => {
                    for &(p, old) in done.iter().rev() {
                        // SAFETY: `p` was just made writable with captured `old`.
                        let _ = unsafe { sys::restore_protection(p, old) };
                    }
                    return Err(err);
                }
            }
        }
        Ok(Self { pages: done })
    }

    /// Restore every page to its captured protection, returning the first error.
    fn restore(self) -> Result<()> {
        let mut first_err = None;
        for &(page, old) in self.pages.iter().rev() {
            // SAFETY: `page` was made writable by this protector with captured `old`.
            if let Err(err) = unsafe { sys::restore_protection(page, old) } {
                first_err.get_or_insert(err);
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Apply a set of `(slot, new_value)` writes transactionally.
///
/// Makes all pages writable, performs the atomic swaps, then restores
/// protections. If making a page writable fails, nothing is swapped. If (against
/// all expectation) restoring protection fails after swapping, the swaps are
/// rolled back before the error is surfaced — so on any error nothing remains
/// patched and no page is left with changed protection.
fn apply(patches: &[(usize, usize)]) -> Result<Vec<usize>> {
    let slots: Vec<usize> = patches.iter().map(|&(slot, _)| slot).collect();
    let protector = PageProtector::make_writable(&slots)?;

    let mut priors = Vec::with_capacity(patches.len());
    for &(slot, new) in patches {
        // SAFETY: every page is writable; slots are validated pointer cells.
        priors.push(unsafe { slot_swap(slot, new) });
    }

    if let Err(err) = protector.restore() {
        // Defensive rollback: undo swaps and best-effort re-restore protections.
        if let Ok(reprotect) = PageProtector::make_writable(&slots) {
            for (&(slot, _), &prior) in patches.iter().zip(&priors) {
                // SAFETY: pages writable again; slots valid.
                unsafe {
                    slot_swap(slot, prior);
                }
            }
            let _ = reprotect.restore();
        }
        return Err(err);
    }
    Ok(priors)
}

/// A single writable import slot discovered in a module.
#[derive(Debug, Clone)]
pub struct ImportSlot {
    module_base: usize,
    library: Arc<str>,
    symbol: Option<Symbol>,
    version: Option<Arc<str>>,
    address: usize,
    kind: ImportKind,
    authenticated: bool,
}

impl ImportSlot {
    pub(crate) fn from_raw(module_base: usize, raw: RawImport) -> Self {
        Self {
            module_base,
            library: raw.library,
            symbol: raw.symbol,
            version: raw.version,
            address: raw.slot,
            kind: raw.kind,
            authenticated: raw.authenticated,
        }
    }

    /// Providing library / DLL name, as recorded in the import descriptor.
    #[must_use]
    pub fn library(&self) -> &str {
        &self.library
    }

    /// The imported symbol, or `None` for a legacy address-only slot
    /// (`OriginalFirstThunk == 0`), whose name is unrecoverable in memory.
    #[must_use]
    pub fn symbol(&self) -> Option<&Symbol> {
        self.symbol.as_ref()
    }

    /// The observed symbol version, e.g. `GLIBC_2.2.5` for an ELF import bound to
    /// unversioned ELF imports. Informational: matching is by name.
    #[must_use]
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// Address of the IAT slot — a `*mut *mut c_void` pointing at the live
    /// function pointer the target dereferences on every call.
    #[must_use]
    pub fn address(&self) -> *mut *mut c_void {
        self.address as *mut *mut c_void
    }

    /// Standard vs delay-load import.
    #[must_use]
    pub fn kind(&self) -> ImportKind {
        self.kind
    }

    /// Whether this slot holds an **authenticated** (PAC-signed) pointer — only on
    /// slot: a raw replacement pointer would fail its `AUTIA`/`AUTDA` check (or, on a
    /// PAC-relaxed CPU, be called unsigned), so the transaction reports
    /// [`Error::AuthenticatedSlot`] rather than corrupt it. Always `false` off
    /// arm64e.
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.authenticated
    }

    /// The current page protection of the slot's IAT page, queried on demand
    /// (informational — the install/restore path captures protection itself).
    ///
    /// Enumeration deliberately does **not** query this per slot: for an image
    /// with many imports that would issue one `VirtualQuery` syscall per slot for
    /// a purely informational value. Callers that want it pay for exactly the
    /// slots they ask about, and always observe the *live* protection rather than
    /// a stale enumeration-time snapshot.
    #[must_use]
    pub fn protection(&self) -> u32 {
        sys::query_protection(self.address)
    }

    /// Base address of the module this slot belongs to.
    #[must_use]
    pub fn module_base(&self) -> *const c_void {
        self.module_base as *const c_void
    }

    /// If this slot satisfies `replacement`, return its matched symbol.
    fn match_replacement(&self, replacement: &Replacement) -> Option<&Symbol> {
        let symbol = self.symbol.as_ref()?;
        if !library_matches(replacement.library.as_deref(), &self.library) {
            return None;
        }
        symbol.matches(&replacement.symbol).then_some(symbol)
    }
}

/// Case-insensitive (ASCII) DLL-name match; `None` filter matches any library.
fn library_matches(filter: Option<&[u8]>, actual: &str) -> bool {
    filter.is_none_or(|wanted| wanted.eq_ignore_ascii_case(actual.as_bytes()))
}

/// A requested slot replacement.
///
/// `library` optionally scopes the match to one providing DLL (case-insensitive);
/// `symbol` names the import; `replacement` is the new function pointer. When
/// `required` is set, [`install`] fails if the symbol is not imported by the
/// module; otherwise a missing symbol is silently skipped — the behavior the
/// `hookfs` install layer needs when applying its superset of candidate hooks.
#[derive(Debug)]
pub struct Replacement {
    /// Optional providing-DLL filter (ASCII, case-insensitive).
    pub library: Option<Vec<u8>>,
    /// The symbol to rebind.
    pub symbol: Symbol,
    /// The replacement function pointer.
    pub replacement: *const c_void,
    /// Fail the install if this symbol is not present (vs. skip it).
    pub required: bool,
}

impl Replacement {
    /// A required by-name replacement in a specific library.
    #[must_use]
    pub fn by_name(library: &str, symbol: &str, replacement: *const c_void) -> Self {
        Self {
            library: Some(library.as_bytes().to_vec()),
            symbol: Symbol::name(symbol),
            replacement,
            required: true,
        }
    }

    /// A required by-name replacement matching the symbol in **any** providing
    /// library — the form the ELF backend uses, where imports are matched by name
    #[must_use]
    pub fn by_symbol(symbol: &str, replacement: *const c_void) -> Self {
        Self {
            library: None,
            symbol: Symbol::name(symbol),
            replacement,
            required: true,
        }
    }

    /// A replacement that is skipped (not an error) if the symbol is absent.
    #[must_use]
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

/// One installed hook: enough to restore it and to hand the caller the original.
#[derive(Debug, Clone)]
pub struct InstalledHook {
    library: Arc<str>,
    symbol: Symbol,
    slot: usize,
    original: usize,
    prior_value: usize,
    replacement: usize,
}

impl InstalledHook {
    /// Providing library of the hooked import.
    #[must_use]
    pub fn library(&self) -> &str {
        &self.library
    }
    /// The hooked symbol.
    #[must_use]
    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }
    /// Address of the patched IAT slot.
    #[must_use]
    pub fn slot(&self) -> *mut *mut c_void {
        self.slot as *mut *mut c_void
    }
    /// The canonical original entry point — the real callee the `hookfs` shims
    /// invoke for passthrough. For a standard import this is `GetProcAddress`
    /// (or the load-time-bound slot value as a fallback); for a delay import it
    /// is always the authoritatively resolved export, never the
    /// `__delayLoadHelper2` stub the delay slot holds before first call.
    #[must_use]
    pub fn original(&self) -> *const c_void {
        self.original as *const c_void
    }
    /// The replacement pointer installed into the slot.
    #[must_use]
    pub fn replacement(&self) -> *const c_void {
        self.replacement as *const c_void
    }
}

/// RAII owner of a set of installed hooks. Dropping it restores the original
/// slot values (best-effort, compare-exchange); [`HookGuard::uninstall`] does the
/// same but surfaces any conflict.
#[derive(Debug)]
pub struct HookGuard {
    module_base: usize,
    hooks: Vec<InstalledHook>,
    active: bool,
}

impl HookGuard {
    /// The hooks this guard installed.
    #[must_use]
    pub fn installed(&self) -> &[InstalledHook] {
        &self.hooks
    }

    /// The canonical original entry point for `symbol`, if this guard hooked it.
    #[must_use]
    pub fn original(&self, symbol: &Symbol) -> Option<*const c_void> {
        self.hooks
            .iter()
            .find(|hook| hook.symbol.matches(symbol))
            .map(InstalledHook::original)
    }

    /// Restore every slot and consume the guard, reporting the first conflict or
    /// protection error encountered (the rest are still restored).
    pub fn uninstall(mut self) -> Result<()> {
        self.restore()
    }

    /// The shared restore path used by both [`Self::uninstall`] and [`Drop`].
    fn restore(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        if self.hooks.is_empty() {
            return Ok(());
        }

        let _lock = install_lock();

        // Liveness: if the module has unloaded, leak the patch state rather than
        if !sys::module_still_loaded(self.module_base) {
            return Ok(());
        }

        let slots: Vec<usize> = self.hooks.iter().map(|hook| hook.slot).collect();
        let protector = PageProtector::make_writable(&slots)?;

        let mut first_err: Option<Error> = None;
        for hook in &self.hooks {
            // Extra guard against base reuse: only touch slots still inside this
            // module's committed image mapping.
            if !sys::slot_belongs_to_module(hook.slot, self.module_base) {
                continue;
            }
            // Compare-exchange: restore only if the slot still holds our
            // SAFETY: page writable; slot validated to belong to the image.
            match unsafe { slot_compare_exchange(hook.slot, hook.replacement, hook.prior_value) } {
                Ok(_) => {}
                Err(found) if found == hook.prior_value => {
                    // Already the original value (benign, e.g. redundant restore).
                }
                Err(found) => {
                    first_err.get_or_insert(Error::RestoreConflict {
                        slot: hook.slot,
                        expected: hook.replacement,
                        found,
                    });
                }
            }
        }

        if let Err(err) = protector.restore() {
            first_err.get_or_insert(err);
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for HookGuard {
    fn drop(&mut self) {
        // Best-effort restore; a Drop must not unwind across into user code.
        let _ = self.restore();
    }
}

struct PendingHook {
    library: Arc<str>,
    symbol: Symbol,
    slot: usize,
    original: usize,
    replacement: usize,
}

/// Install a set of [`Replacement`]s into `module`'s import table, transactionally.
///
/// All target slots are resolved first (and each `required` symbol verified
/// present); all originals are resolved before any slot is written; the writes
/// are then applied atomically with page-batched protection changes and rolled
/// restores the originals when dropped or [`HookGuard::uninstall`]ed.
pub fn install(module: &Module, replacements: &[Replacement]) -> Result<HookGuard> {
    let _lock = install_lock();

    let imports = module.imports()?;
    let mut pending: Vec<PendingHook> = Vec::new();

    for replacement in replacements {
        let mut matched = false;
        for slot in &imports {
            let Some(symbol) = slot.match_replacement(replacement) else {
                continue;
            };
            matched = true;
            // a correctly PAC-signed replacement we cannot synthesize. Refuse loudly
            // rather than write a pointer that fails its auth check (or, worse, is
            // dereferenced unsigned) — fail-closed beats a corrupt slot. This aborts
            // the whole transaction before any write, so nothing is left patched.
            if slot.authenticated {
                return Err(Error::AuthenticatedSlot {
                    module: module.name().to_owned(),
                    symbol: symbol.describe(),
                });
            }
            if pending.iter().any(|hook| hook.slot == slot.address) {
                continue;
            }
            // Resolve the canonical original before any write (R6). It must be the
            // real callee for passthrough — never a resolver stub. The per-platform
            // policy lives in `sys::resolve_original`: PE prefers `GetProcAddress`
            // with the load-time slot value as a fallback (delay imports force-load
            // the provider); ELF always uses `dlsym(RTLD_DEFAULT)` and never the
            // lazily-bound slot. A `None` here means the original is genuinely
            // unresolvable — fail a `required` hook rather than hand out a stub, or
            // skip an `optional` one.
            // SAFETY: `slot.address` is a readable, pointer-aligned slot cell.
            let slot_value = unsafe { slot_load(slot.address) };
            let original = match sys::resolve_original(slot.kind, &slot.library, symbol, slot_value)
            {
                Some(original) => original,
                None if replacement.required => {
                    return Err(Error::OriginalUnresolved {
                        module: module.name().to_owned(),
                        library: slot.library.to_string(),
                        symbol: symbol.describe(),
                    });
                }
                None => continue,
            };
            pending.push(PendingHook {
                library: slot.library.clone(),
                symbol: symbol.clone(),
                slot: slot.address,
                original,
                replacement: replacement.replacement as usize,
            });
        }
        if !matched && replacement.required {
            return Err(Error::SymbolNotFound {
                module: module.name().to_owned(),
                symbol: replacement.symbol.describe(),
            });
        }
    }

    if pending.is_empty() {
        return Ok(HookGuard {
            module_base: module.base_addr(),
            hooks: Vec::new(),
            active: true,
        });
    }

    let patches: Vec<(usize, usize)> = pending
        .iter()
        .map(|hook| (hook.slot, hook.replacement))
        .collect();
    let priors = apply(&patches)?;

    let hooks = pending
        .into_iter()
        .zip(priors)
        .map(|(hook, prior_value)| InstalledHook {
            library: hook.library,
            symbol: hook.symbol,
            slot: hook.slot,
            original: hook.original,
            prior_value,
            replacement: hook.replacement,
        })
        .collect();

    Ok(HookGuard {
        module_base: module.base_addr(),
        hooks,
        active: true,
    })
}

/// `Replacement` holds a raw function pointer, which is `!Send` by default; the
/// pointer is a process-global code address safe to move between threads.
// SAFETY: the pointer refers to a `'static` function in a loaded module.
unsafe impl Send for Replacement {}
// SAFETY: `Replacement` exposes the pointer only by value; no interior mutability.
unsafe impl Sync for Replacement {}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unnecessary_box_returns,
    clippy::borrowed_box,
    clippy::ref_as_ptr
)]
mod tests {
    use super::*;

    // A heap-allocated pointer-sized cell standing in for an import slot. Its
    // page is ordinary committed read/write memory, so `VirtualProtect` succeeds
    // — exactly what the transaction needs to exercise its protection dance.
    fn cell(value: usize) -> Box<usize> {
        Box::new(value)
    }
    fn addr(cell: &Box<usize>) -> usize {
        cell.as_ref() as *const usize as usize
    }

    #[test]
    fn apply_swaps_and_reports_priors() {
        let slot = cell(0x1111);
        let priors = apply(&[(addr(&slot), 0x2222)]).expect("apply succeeds");
        assert_eq!(priors, vec![0x1111]);
        assert_eq!(*slot, 0x2222);
    }

    #[test]
    fn apply_rolls_back_on_injected_failure() {
        let good = cell(0xAAAA);
        // Second "slot" is a bogus, unmapped address: making its page writable
        // fails, so the whole transaction must abort with nothing swapped.
        let patches = [(addr(&good), 0xBBBB), (0x1usize, 0xCCCC)];
        let result = apply(&patches);
        assert!(matches!(result, Err(Error::Protect { .. })));
        // The good slot must be untouched (no swap happened before the failure).
        assert_eq!(*good, 0xAAAA);
    }

    #[test]
    fn compare_exchange_restore_detects_a_subsequent_hook() {
        let slot = cell(0x1000); // original
        let a = addr(&slot);
        // Our hook: original -> replacement A.
        let priors = apply(&[(a, 0x2000)]).expect("install A");
        assert_eq!(priors, vec![0x1000]);

        // SAFETY: heap cell, writable, aligned.
        unsafe { slot_swap(a, 0x3000) };

        // Restoring A with compare-exchange must NOT clobber B: the slot no
        // longer holds A's replacement (0x2000), so the exchange fails.
        // SAFETY: heap cell, writable, aligned.
        let result = unsafe { slot_compare_exchange(a, 0x2000, 0x1000) };
        assert_eq!(result, Err(0x3000));
        assert_eq!(*slot, 0x3000, "subsequent hook preserved");
    }

    #[test]
    fn library_matching_is_case_insensitive_and_optional() {
        assert!(library_matches(None, "KERNEL32.dll"));
        assert!(library_matches(Some(b"kernel32.DLL"), "KERNEL32.dll"));
        assert!(!library_matches(Some(b"ole32.dll"), "KERNEL32.dll"));
    }
}
