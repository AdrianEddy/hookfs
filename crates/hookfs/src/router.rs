//! The router: the per-call virtual-vs-real decision, relative-path resolution
//! against a tracked virtual cwd, and the [`Engine`] that bundles the active
//!
//! The reserved-prefix invariant lives here: a path that *lexically* falls under
//! the virtual prefix is either served by the VFS or fails closed — it is never
//! canonicalized against, or allowed to fall through to, the physical filesystem.

#[cfg(hookfs_backend)]
use crate::namespace::{Namespace, is_lexically_safe};
#[cfg(hookfs_backend)]
use crate::registry::Registry;
#[cfg(hookfs_backend)]
use crate::vfs::VirtualFs;
#[cfg(hookfs_backend)]
use std::path::{Path, PathBuf};
#[cfg(hookfs_backend)]
use std::sync::{Arc, Mutex};

/// The decision for one path-bearing call.
#[cfg(hookfs_backend)]
#[derive(Debug)]
pub(crate) enum Route {
    /// A safe virtual path to service from the VFS (normalized display form).
    Virtual(PathBuf),
    /// A virtual-prefix path that is syntactically unsafe (`..` escape, embedded
    /// NUL, stream/device marker): fail closed, never touch disk.
    Rejected,
    /// Not a virtual path: pass straight through to the real OS.
    Real,
}

/// The active routing state, published globally while an installation is live.
#[cfg(hookfs_backend)]
pub(crate) struct Engine {
    vfs: Arc<dyn VirtualFs>,
    namespace: Namespace,
    registry: Registry,
    /// The tracked virtual current directory (display form), used to resolve
    /// relative paths. `None` = fall back to the real cwd (relative real paths
    /// then pass through untouched).
    cwd: Mutex<Option<String>>,
    allow_writes: bool,
}

#[cfg(hookfs_backend)]
impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("namespace", &self.namespace)
            .field("registry", &self.registry)
            .field("allow_writes", &self.allow_writes)
            .finish_non_exhaustive()
    }
}

#[cfg(hookfs_backend)]
impl Engine {
    /// Build an engine over `vfs`, `namespace`, and a fresh registry.
    pub(crate) fn new(vfs: Arc<dyn VirtualFs>, namespace: Namespace, allow_writes: bool) -> Self {
        Self {
            vfs,
            namespace,
            registry: Registry::new(),
            cwd: Mutex::new(None),
            allow_writes,
        }
    }

    /// The active virtual filesystem.
    pub(crate) fn vfs(&self) -> &Arc<dyn VirtualFs> {
        &self.vfs
    }

    /// The open-handle registry.
    pub(crate) fn registry(&self) -> &Registry {
        &self.registry
    }

    pub(crate) fn allow_writes(&self) -> bool {
        self.allow_writes
    }

    /// The current virtual cwd, if set.
    pub(crate) fn virtual_cwd(&self) -> Option<String> {
        self.cwd
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Set (or clear) the virtual cwd, for `chdir`/`SetCurrentDirectoryW` tracking
    /// yet (the SDK never imports `SetCurrentDirectoryW`), so on Windows it exists
    /// only for the router's own tests. Gated to where a caller exists rather than
    /// blanket-silenced.
    #[cfg(any(test, unix))]
    pub(crate) fn set_virtual_cwd(&self, cwd: Option<String>) {
        *self
            .cwd
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = cwd;
    }

    /// Classify a decoded path string into a [`Route`], resolving a relative path
    /// against the virtual cwd first.
    pub(crate) fn classify(&self, decoded: &str) -> Route {
        // Resolve relative paths.
        let absolute: PathBuf = if is_absolute(decoded) {
            PathBuf::from(decoded)
        } else if let Some(cwd) = self.virtual_cwd() {
            Path::new(&cwd).join(decoded)
        } else {
            // A relative path with no virtual cwd is a real relative path.
            return Route::Real;
        };

        if !self.namespace.is_virtual(&absolute) {
            return Route::Real;
        }
        if is_lexically_safe(&absolute.to_string_lossy()) {
            Route::Virtual(absolute)
        } else {
            Route::Rejected
        }
    }
}

/// Whether a decoded path string is absolute.
///
/// Windows: a drive-absolute (`C:\`), a UNC / device (`\\`), or a long-path
/// (`\\?\`) form. POSIX: a leading `/`.
#[cfg(windows)]
pub(crate) fn is_absolute(s: &str) -> bool {
    if s.starts_with("\\\\") || s.starts_with("//") {
        return true;
    }
    let bytes = s.as_bytes();
    // Drive-absolute: `<letter>:` followed by a separator.
    bytes.first().is_some_and(u8::is_ascii_alphabetic)
        && bytes.get(1) == Some(&b':')
        && matches!(bytes.get(2), Some(b'\\' | b'/'))
}

/// POSIX: an absolute path begins with `/`.
#[cfg(unix)]
pub(crate) fn is_absolute(s: &str) -> bool {
    s.starts_with('/')
}

#[cfg(all(test, windows))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::providers::MemoryFs;

    fn engine() -> (Engine, PathBuf) {
        let root = PathBuf::from("C:\\__hookfs__\\rtest");
        (
            Engine::new(MemoryFs::new(), Namespace::from_root(root.clone()), false),
            root,
        )
    }

    #[test]
    fn absoluteness() {
        assert!(is_absolute("C:\\a"));
        assert!(is_absolute("c:/a"));
        assert!(is_absolute("\\\\server\\share"));
        assert!(is_absolute("\\\\?\\C:\\a"));
        assert!(!is_absolute("a\\b"));
        assert!(!is_absolute("relative"));
    }

    #[test]
    fn classifies_virtual_real_and_rejected() {
        let (e, root) = engine();
        let clip = root.join("sample.braw");
        assert!(matches!(
            e.classify(&clip.to_string_lossy()),
            Route::Virtual(_)
        ));
        assert!(matches!(
            e.classify("C:\\Windows\\notepad.exe"),
            Route::Real
        ));

        let escape = format!("{}\\..\\..\\secret", root.display());
        assert!(matches!(e.classify(&escape), Route::Rejected));

        // A relative path with no virtual cwd is real.
        assert!(matches!(e.classify("sample.braw"), Route::Real));
    }

    #[test]
    fn relative_paths_resolve_against_virtual_cwd() {
        let (e, root) = engine();
        e.set_virtual_cwd(Some(root.to_string_lossy().into_owned()));
        assert!(matches!(e.classify("sample.braw"), Route::Virtual(_)));
    }
}

#[cfg(all(test, unix))]
#[allow(clippy::unwrap_used)]
mod posix_tests {
    use super::*;
    #[cfg(hookfs_backend)]
    use crate::providers::MemoryFs;

    // The routing `Engine` exists only on a target with a shim backend; its POSIX
    // tests are gated to match, while the pure `is_absolute` helper is exercised on
    // every unix target.
    #[cfg(hookfs_backend)]
    fn engine() -> (Engine, PathBuf) {
        let root = PathBuf::from("/__hookfs__/rtest");
        (
            Engine::new(MemoryFs::new(), Namespace::from_root(root.clone()), false),
            root,
        )
    }

    #[test]
    fn absoluteness() {
        assert!(is_absolute("/etc/passwd"));
        assert!(!is_absolute("relative/path"));
        assert!(!is_absolute("./a"));
    }

    #[cfg(hookfs_backend)]
    #[test]
    fn classifies_virtual_real_and_rejected() {
        let (e, root) = engine();
        let clip = root.join("sample.braw");
        assert!(matches!(
            e.classify(&clip.to_string_lossy()),
            Route::Virtual(_)
        ));
        assert!(matches!(e.classify("/usr/bin/ls"), Route::Real));

        let escape = format!("{}/../../secret", root.display());
        assert!(matches!(e.classify(&escape), Route::Rejected));

        // A relative path with no virtual cwd is real.
        assert!(matches!(e.classify("sample.braw"), Route::Real));
    }

    #[cfg(hookfs_backend)]
    #[test]
    fn relative_paths_resolve_against_virtual_cwd() {
        let (e, root) = engine();
        e.set_virtual_cwd(Some(root.to_string_lossy().into_owned()));
        assert!(matches!(e.classify("sample.braw"), Route::Virtual(_)));
    }
}
