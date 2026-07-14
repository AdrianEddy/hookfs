//! Hook installation: pick the symbols, resolve the originals, patch the scoped
//! modules through `plthook`, publish the routing engine, and support rescanning

use crate::error::{Error, Result};
use crate::namespace::Namespace;
use crate::vfs::VirtualFs;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(hookfs_backend)]
use crate::dispatch::{self, Sym};
#[cfg(hookfs_backend)]
use crate::router::Engine;
#[cfg(hookfs_backend)]
use std::sync::Mutex;

/// Which modules to patch. The BRAW default is [`Scope::Module`] — the single
#[derive(Debug, Clone)]
pub enum Scope {
    /// Patch only the module containing this address (stored as `usize` so the
    /// option is `Send`). Discover it from, e.g., `CreateBlackmagicRawFactoryInstance`.
    Module(usize),
    /// Patch every module currently loaded.
    AllModules,
    /// Patch only modules whose file name matches (case-insensitive).
    Only(Vec<String>),
    /// Patch every module except those whose file name matches.
    Exclude(Vec<String>),
}

/// How a path is judged virtual.
enum Matcher {
    /// The reserved-prefix default (or `for_prefix`) — supplied by the caller as a
    /// concrete [`Namespace`].
    Namespace(Namespace),
    /// A fully custom predicate over the reserved root.
    Predicate(Box<dyn Fn(&std::path::Path) -> bool + Send + Sync>),
}

pub struct Options {
    scope: Scope,
    allow_writes: bool,
    auto_rescan: bool,
    matcher: Option<Matcher>,
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("scope", &self.scope)
            .field("allow_writes", &self.allow_writes)
            .field("auto_rescan", &self.auto_rescan)
            .field("custom_matcher", &self.matcher.is_some())
            .finish()
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            scope: Scope::AllModules,
            allow_writes: false,
            auto_rescan: false,
            matcher: None,
        }
    }
}

impl Options {
    /// Options that patch the module containing `address` (the SDK module,
    /// discovered by the address of an exported function).
    #[must_use]
    pub fn for_module(address: *const core::ffi::c_void) -> Self {
        Self {
            scope: Scope::Module(address as usize),
            ..Self::default()
        }
    }

    /// Options that treat any path under `prefix` as virtual (a custom reserved
    /// prefix instead of the process-random default).
    #[must_use]
    pub fn for_prefix(prefix: impl Into<PathBuf>) -> Self {
        Self {
            matcher: Some(Matcher::Namespace(Namespace::for_prefix(prefix))),
            ..Self::default()
        }
    }

    /// Set the module scope.
    #[must_use]
    pub fn scope(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    #[must_use]
    pub fn allow_writes(mut self, allow: bool) -> Self {
        self.allow_writes = allow;
        self
    }

    /// Re-run installation on late-loaded modules by hooking the loader
    /// (`LoadLibrary*`), so decoder plugins are patched too (R7).
    #[must_use]
    pub fn auto_rescan(mut self, enable: bool) -> Self {
        self.auto_rescan = enable;
        self
    }

    /// Supply a fully custom virtual-path predicate.
    #[must_use]
    pub fn is_virtual(
        mut self,
        predicate: impl Fn(&std::path::Path) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.matcher = Some(Matcher::Predicate(Box::new(predicate)));
        self
    }

    /// Decompose into the parts the convenience mount layer needs (its own
    /// namespace supersedes any matcher here).
    pub(crate) fn into_parts(self) -> (Scope, bool, bool) {
        (self.scope, self.allow_writes, self.auto_rescan)
    }
}

/// Install `hookfs` over `vfs` per `opts`, returning a guard that restores the
/// original imports when dropped.
///
/// The caller supplies a [`VirtualFs`] whose paths it already knows (for the
/// reserved-prefix convenience form, drive it through
/// [`Hookfs`](crate::Hookfs) instead, which knows the random prefix).
///
/// # Single active installation
/// `install` while an [`InstallGuard`] is still live is refused with
/// [`Error::AlreadyInstalled`] rather than silently hijacking the first. Drop the
/// existing guard to tear the installation down before installing again.
///
/// # Errors
/// - [`Error::AlreadyInstalled`] if an installation is already active in this
///   process.
/// - [`Error::NoTargetModule`] if the scope resolves to no loaded module.
/// - [`Error::Engine`] if a target image cannot be parsed or patched.
pub fn install(vfs: Arc<dyn VirtualFs>, opts: Options) -> Result<InstallGuard> {
    let namespace = match opts.matcher {
        Some(Matcher::Namespace(ns)) => ns,
        Some(Matcher::Predicate(pred)) => Namespace::with_predicate(pred),
        None => Namespace::reserved(),
    };
    install_with_namespace(
        vfs,
        namespace,
        opts.scope,
        opts.allow_writes,
        opts.auto_rescan,
    )
}

// ---------------------------------------------------------------------------
// Shared install machinery (Windows PE + Linux ELF)
// ---------------------------------------------------------------------------

/// Install with an explicit namespace (used by the convenience mount layer, which
/// owns the reserved namespace it also mounts into).
#[cfg(hookfs_backend)]
pub(crate) fn install_with_namespace(
    vfs: Arc<dyn VirtualFs>,
    namespace: Namespace,
    scope: Scope,
    allow_writes: bool,
    auto_rescan: bool,
) -> Result<InstallGuard> {
    let modules = resolve_modules(&scope)?;
    if modules.is_empty() {
        return Err(Error::NoTargetModule("scope matched no loaded module"));
    }

    // publish this engine as THE active routing state *before* patching, so a shim
    // that runs immediately after a slot is swapped already sees it — or fail
    // loudly if an installation is already live rather than silently clobbering the
    // shared global engine that both installations would then fight over.
    let engine = Arc::new(Engine::new(vfs, namespace, allow_writes));
    let install_id = dispatch::try_activate(engine.clone()).ok_or(Error::AlreadyInstalled)?;

    // Resolve every candidate original from KERNEL32 up front, before any slot is
    // patched. This both closes the race (a hook can never run with a null
    // original) and guarantees the carrier-handle helpers work even in a module
    // that hooks CreateFile2 but not CreateFileW.
    preresolve_originals();

    let replacements = build_replacements(auto_rescan);
    let sdk_dir = anchor_dir(&scope);

    let installer = Arc::new(Installer {
        scope,
        sdk_dir,
        auto_rescan,
        install_id,
        guards: Mutex::new(Vec::new()),
        hooked_bases: Mutex::new(std::collections::HashSet::new()),
        _engine: engine,
    });

    // Install into each targeted module. On any failure, roll back everything.
    for module in &modules {
        match plthook::install(module, &replacements) {
            Ok(guard) => {
                refresh_originals(&guard);
                installer
                    .hooked_bases
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(module.base() as usize);
                installer
                    .guards
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(guard);
            }
            Err(err) => {
                // Dropping the collected guards restores the slots; then relinquish
                // this installation's ownership of the global routing state.
                installer.uninstall_all();
                dispatch::deactivate(install_id);
                return Err(Error::Engine(err));
            }
        }
    }

    if auto_rescan {
        let installer_for_cb = installer.clone();
        dispatch::set_rescan(Arc::new(move || {
            let _ = installer_for_cb.rescan();
        }));
    }

    Ok(InstallGuard { installer })
}

/// The engine + guards + rescan state; shared by [`InstallGuard`] and the loader
/// rescan callback so both add to and tear down the same guard set.
#[cfg(hookfs_backend)]
struct Installer {
    scope: Scope,
    /// Directory of the anchor module (lowercased key), for SDK-sibling rescans.
    sdk_dir: Option<String>,
    auto_rescan: bool,
    /// This installation's unique token in the global single-active slot; teardown
    install_id: u64,
    guards: Mutex<Vec<plthook::HookGuard>>,
    hooked_bases: Mutex<std::collections::HashSet<usize>>,
    /// Keeps the engine alive for as long as the installation exists.
    _engine: Arc<Engine>,
}

#[cfg(hookfs_backend)]
impl Installer {
    /// Re-run installation on any not-yet-hooked module that this scope covers.
    fn rescan(&self) -> Result<usize> {
        let replacements = build_replacements(self.auto_rescan);
        let modules = plthook::Module::enumerate().map_err(Error::Engine)?;
        let mut count = 0;
        for module in modules {
            let base = module.base() as usize;
            if !self.should_rescan_hook(&module) {
                continue;
            }
            {
                let mut bases = self
                    .hooked_bases
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if bases.contains(&base) {
                    continue;
                }
                bases.insert(base);
            }
            if let Ok(guard) = plthook::install(&module, &replacements) {
                refresh_originals(&guard);
                self.guards
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(guard);
                count += 1;
            }
        }
        Ok(count)
    }

    /// Whether a late module should be hooked on rescan.
    fn should_rescan_hook(&self, module: &plthook::Module) -> bool {
        match &self.scope {
            // Hook SDK-directory siblings of the anchor module.
            Scope::Module(_) => self
                .sdk_dir
                .as_deref()
                .is_some_and(|dir| module_dir_key(module) == dir),
            Scope::AllModules => true,
            Scope::Only(names) => names.iter().any(|n| name_eq(module.name(), n)),
            Scope::Exclude(names) => !names.iter().any(|n| name_eq(module.name(), n)),
        }
    }

    /// Uninstall every collected guard (best-effort, compare-exchange restore).
    fn uninstall_all(&self) {
        let guards = std::mem::take(
            &mut *self
                .guards
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for guard in guards {
            let _ = guard.uninstall();
        }
    }
}

/// RAII owner of an installation. Dropping it restores the original imports
/// (compare-exchange, best-effort) and clears the active routing state.
#[cfg(hookfs_backend)]
pub struct InstallGuard {
    installer: Arc<Installer>,
}

#[cfg(hookfs_backend)]
impl std::fmt::Debug for InstallGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hooked = self.installer.hooked_bases.lock().map_or(0, |b| b.len());
        f.debug_struct("InstallGuard")
            .field("hooked_modules", &hooked)
            .finish()
    }
}

#[cfg(hookfs_backend)]
impl InstallGuard {
    /// Re-run installation on late-loaded modules (e.g. decoder plugins). Returns
    /// the number of newly-hooked modules.
    ///
    /// # Errors
    /// Propagates a module-enumeration failure from the engine.
    pub fn rescan(&self) -> Result<usize> {
        self.installer.rescan()
    }

    /// The number of modules currently hooked by this installation.
    #[must_use]
    pub fn hooked_module_count(&self) -> usize {
        self.installer.hooked_bases.lock().map_or(0, |b| b.len())
    }
}

#[cfg(hookfs_backend)]
impl Drop for InstallGuard {
    fn drop(&mut self) {
        dispatch::clear_rescan();
        self.installer.uninstall_all();
        // Relinquish the global routing state only if this installation still owns
        // it — a no-op otherwise, so teardown can never clear another installation.
        dispatch::deactivate(self.installer.install_id);
    }
}

// ---- Platform helpers ------------------------------------------------------

/// Build the optional (skip-if-absent) replacement set. The loader shims are
/// included only when `auto_rescan` is on.
#[cfg(windows)]
fn build_replacements(auto_rescan: bool) -> Vec<plthook::Replacement> {
    use crate::shims::windows::shim_address;
    dispatch::ALL
        .iter()
        .filter(|&&sym| auto_rescan || !matches!(sym, Sym::LoadLibraryExW | Sym::LoadLibraryA))
        .map(|&sym| {
            plthook::Replacement::by_name("KERNEL32.dll", sym.name(), shim_address(sym)).optional()
        })
        .collect()
}

/// Build the optional replacement set for the ELF (Linux) and Mach-O (Darwin: macOS
/// and iOS/iPadOS) backends. Imports are matched by symbol name in **any** providing
/// library (there is no per-slot library filter on ELF, and the Mach-O parser
/// normalizes names to the same base form), and the loader shims (`dlopen`/`dlclose`)
/// are included only when `auto_rescan` is on. The per-platform `Sym` set and
/// `shim_address` differ; the wiring is identical.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
fn build_replacements(auto_rescan: bool) -> Vec<plthook::Replacement> {
    use crate::shims::unix::shim_address;
    dispatch::ALL
        .iter()
        .filter(|&&sym| auto_rescan || !matches!(sym, Sym::Dlopen | Sym::Dlclose))
        .map(|&sym| plthook::Replacement::by_symbol(sym.name(), shim_address(sym)).optional())
        .collect()
}

/// Resolve the scope to a concrete list of modules.
#[cfg(hookfs_backend)]
fn resolve_modules(scope: &Scope) -> Result<Vec<plthook::Module>> {
    match scope {
        Scope::Module(addr) => {
            // SAFETY: the address was produced from a live export of the target
            // module by the caller (e.g. `CreateBlackmagicRawFactoryInstance`).
            let module =
                unsafe { plthook::Module::from_address(*addr as *const core::ffi::c_void) }
                    .map_err(Error::Engine)?;
            Ok(vec![module])
        }
        Scope::AllModules => plthook::Module::enumerate().map_err(Error::Engine),
        Scope::Only(names) => Ok(plthook::Module::enumerate()
            .map_err(Error::Engine)?
            .into_iter()
            .filter(|m| names.iter().any(|n| name_eq(m.name(), n)))
            .collect()),
        Scope::Exclude(names) => Ok(plthook::Module::enumerate()
            .map_err(Error::Engine)?
            .into_iter()
            .filter(|m| !names.iter().any(|n| name_eq(m.name(), n)))
            .collect()),
    }
}

/// The directory key of the anchor module for a `Scope::Module`, else `None`.
#[cfg(hookfs_backend)]
fn anchor_dir(scope: &Scope) -> Option<String> {
    match scope {
        Scope::Module(addr) => {
            // SAFETY: caller-provided live export address.
            let module =
                unsafe { plthook::Module::from_address(*addr as *const core::ffi::c_void) }.ok()?;
            Some(module_dir_key(&module))
        }
        _ => None,
    }
}

/// The normalized directory key (lowercase) of a module's on-disk path.
#[cfg(hookfs_backend)]
fn module_dir_key(module: &plthook::Module) -> String {
    let path = std::path::Path::new(module.path());
    let dir = path.parent().unwrap_or(path);
    crate::namespace::normalize_key(dir)
}

/// Case-insensitive module-name comparison.
#[cfg(hookfs_backend)]
fn name_eq(actual: &str, wanted: &str) -> bool {
    actual.eq_ignore_ascii_case(wanted)
}

/// Resolve every candidate original from `KERNEL32.dll` and record it, before any
/// slot is patched.
#[cfg(windows)]
fn preresolve_originals() {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    let kernel32: Vec<u16> = "kernel32.dll\0".encode_utf16().collect();
    // SAFETY: valid NUL-terminated wide name; KERNEL32 is always loaded.
    let module = unsafe { GetModuleHandleW(kernel32.as_ptr()) };
    if module.is_null() {
        return;
    }
    for sym in dispatch::ALL {
        let mut cname: Vec<u8> = sym.name().as_bytes().to_vec();
        cname.push(0);
        // SAFETY: `cname` is a valid NUL-terminated name; `module` is live.
        if let Some(proc) = unsafe { GetProcAddress(module, cname.as_ptr()) } {
            dispatch::set_original(sym.name(), proc as usize);
        }
    }
}

/// Resolve every candidate original from libc via `dlsym(RTLD_DEFAULT, name)` and
/// record it, before any slot is patched. This closes the race (a hook can never
/// run with a null original) and guarantees passthrough never discovers an original
/// recursively — including the carrier-fd / cookie-stream helpers, which call the
/// (macOS + iOS): `dlsym(RTLD_DEFAULT, base_name)`; on Darwin `dlsym` re-adds the
/// Mach-O leading `_`, and the base names match the parser's normalized form.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
fn preresolve_originals() {
    for sym in dispatch::ALL {
        let mut cname: Vec<u8> = sym.name().as_bytes().to_vec();
        cname.push(0);
        // SAFETY: `cname` is a valid NUL-terminated C string; `RTLD_DEFAULT`
        // performs the default global lookup, returning the canonical libc export.
        let proc = unsafe {
            libc::dlsym(
                libc::RTLD_DEFAULT,
                cname.as_ptr().cast::<core::ffi::c_char>(),
            )
        };
        if !proc.is_null() {
            dispatch::set_original(sym.name(), proc as usize);
        }
    }
}

/// Refresh the originals table from a freshly-installed guard (the canonical
/// entry points the engine resolved).
#[cfg(hookfs_backend)]
fn refresh_originals(guard: &plthook::HookGuard) {
    for hook in guard.installed() {
        dispatch::set_original(&hook.symbol().describe(), hook.original() as usize);
    }
}

#[cfg(all(windows, test))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::providers::MemoryFs;
    use crate::vfs::VirtualFs;

    /// A minimal real scope: patch the module containing this test binary's own
    /// code (its address). That gives a genuine, in-process installation to prove
    /// the single-active enforcement through the public `install` entry point.
    fn self_scope() -> Scope {
        Scope::Module(self_scope as *const core::ffi::c_void as usize)
    }

    #[test]
    fn second_install_is_rejected_until_the_first_guard_drops() {
        let _serial = dispatch::GLOBAL_STATE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let vfs: Arc<dyn VirtualFs> = MemoryFs::new();

        // First installation succeeds and holds the single active slot.
        let first = install(vfs.clone(), Options::default().scope(self_scope()))
            .expect("first install should succeed");

        // A second install while the first guard is live is refused (no clobber).
        let err = install(vfs.clone(), Options::default().scope(self_scope()))
            .expect_err("second install must be rejected while one is active");
        assert!(
            matches!(err, Error::AlreadyInstalled),
            "unexpected error: {err:?}"
        );

        // After the first guard drops, installing succeeds again.
        drop(first);
        let second = install(vfs, Options::default().scope(self_scope()))
            .expect("install should succeed again after teardown");
        drop(second);
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[cfg(not(hookfs_backend))]
pub(crate) fn install_with_namespace(
    _vfs: Arc<dyn VirtualFs>,
    _namespace: Namespace,
    _scope: Scope,
    _allow_writes: bool,
    _auto_rescan: bool,
) -> Result<InstallGuard> {
    Err(Error::UnsupportedPlatform)
}

#[cfg(not(hookfs_backend))]
#[derive(Debug)]
pub struct InstallGuard {
    _private: (),
}

#[cfg(not(hookfs_backend))]
impl InstallGuard {
    /// This target has no shim backend.
    ///
    /// # Errors
    /// Always returns [`Error::UnsupportedPlatform`].
    pub fn rescan(&self) -> Result<usize> {
        Err(Error::UnsupportedPlatform)
    }
}
