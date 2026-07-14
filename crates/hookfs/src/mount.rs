//! `Read + Seek` streams under a reserved synthetic path, then install the hooks.
//!
//! Every mount gives each native open an **independent logical cursor** over the
//! shared source (via [`MemoryFs`]). Dropping a [`MountGuard`] removes the name
//! cursor holds its own `Arc`.

use crate::error::{Error, Result};
use crate::install::{InstallGuard, Options};
use crate::namespace::{Namespace, is_lexically_safe, reserved_root};
use crate::providers::MemoryFs;
use crate::router::is_absolute;
use crate::vfs::{FileStream, SharedSource, VirtualFs};
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A mount context: a reserved synthetic directory plus the in-memory tree that
/// backs it. Mount streams into it, then [`install`](Hookfs::install) the hooks.
#[derive(Clone, Debug)]
pub struct Hookfs {
    fs: Arc<MemoryFs>,
    root: PathBuf,
}

impl Default for Hookfs {
    fn default() -> Self {
        Self::new()
    }
}

impl Hookfs {
    /// A fresh context with a process-random reserved directory and the default
    /// (1 TiB) writable capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fs: MemoryFs::new(),
            root: reserved_root(),
        }
    }

    /// A fresh context whose writable volume holds at most `capacity` bytes; writes
    /// with [`Options::allow_writes`] to permit writes at all.
    #[must_use]
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            fs: MemoryFs::with_capacity(capacity),
            root: reserved_root(),
        }
    }

    /// The reserved synthetic directory (`C:\__hookfs__\<id>`).
    #[must_use]
    pub fn virtual_dir(&self) -> &Path {
        &self.root
    }

    /// The synthetic absolute path a `logical_name` maps to.
    #[must_use]
    pub fn path_for(&self, logical_name: &str) -> PathBuf {
        self.root.join(logical_name)
    }

    /// The underlying provider (e.g. to compose with an [`OverlayFs`](crate::OverlayFs)).
    #[must_use]
    pub fn vfs(&self) -> Arc<dyn VirtualFs> {
        self.fs.clone()
    }

    /// Mount `stream` under `logical_name` (a relative name/sub-path). Returns a
    /// guard that unmounts the name when dropped.
    ///
    /// # Errors
    /// [`Error::InvalidMountName`] for an unsafe name, or [`Error::StreamSize`] if
    /// the stream's length can't be probed.
    pub fn mount<R: Read + Seek + Send + 'static>(
        &self,
        logical_name: &str,
        stream: R,
    ) -> Result<MountGuard> {
        validate_logical(logical_name).map_err(|reason| Error::InvalidMountName {
            name: logical_name.to_owned(),
            reason,
        })?;
        let abs = self.root.join(logical_name);
        let source = SharedSource::new(stream).map_err(|source| Error::StreamSize {
            name: logical_name.to_owned(),
            source,
        })?;
        let key = self.fs.insert_file(&abs, source, None);
        Ok(MountGuard {
            fs: self.fs.clone(),
            key,
        })
    }

    /// Mount an already-boxed [`FileStream`] (used by callers that erase the
    /// reader type up front, e.g. to hold several heterogeneous siblings).
    ///
    /// # Errors
    /// As [`Hookfs::mount`].
    pub fn mount_boxed(
        &self,
        logical_name: &str,
        stream: Box<dyn FileStream>,
    ) -> Result<MountGuard> {
        self.mount(logical_name, BoxedStream(stream))
    }

    /// growable file the write shims can write, truncate, and read back through the
    /// same synthetic path (the sidecar the SDK saves, a trim job's output). Returns
    /// a guard that unmounts (and frees) it when dropped.
    ///
    /// Installing with [`Options::allow_writes`] is required for writes to reach it;
    /// a writable file also appears in directory enumeration.
    ///
    /// # Errors
    /// [`Error::InvalidMountName`] for an unsafe name.
    pub fn mount_writable(&self, logical_name: &str) -> Result<MountGuard> {
        validate_logical(logical_name).map_err(|reason| Error::InvalidMountName {
            name: logical_name.to_owned(),
            reason,
        })?;
        let abs = self.root.join(logical_name);
        let key = self.fs.insert_writable(&abs);
        Ok(MountGuard {
            fs: self.fs.clone(),
            key,
        })
    }

    /// A byte-for-byte copy of the current contents of the writable file at
    /// `logical_name`, or `None` if it is not a writable in-memory file. Lets a
    /// caller read back what a native writer produced without touching disk.
    #[must_use]
    pub fn read_virtual(&self, logical_name: &str) -> Option<Vec<u8>> {
        self.fs.snapshot(&self.root.join(logical_name))
    }

    /// A byte-for-byte copy of the writable file at an absolute synthetic `path`
    /// (e.g. one created by the SDK at a path derived from the clip's location).
    #[must_use]
    pub fn read_virtual_path(&self, path: &Path) -> Option<Vec<u8>> {
        self.fs.snapshot(path)
    }

    /// Install the hooks for this context, using its reserved namespace.
    ///
    /// # Errors
    /// As [`crate::install`].
    pub fn install(&self, opts: Options) -> Result<InstallGuard> {
        let namespace = Namespace::from_root(self.root.clone());
        let (scope, allow_writes, auto_rescan) = opts.into_parts();
        crate::install::install_with_namespace(
            self.fs.clone(),
            namespace,
            scope,
            allow_writes,
            auto_rescan,
        )
    }
}

/// Removes a mounted name when dropped. Already-open handles are unaffected.
#[derive(Debug)]
pub struct MountGuard {
    fs: Arc<MemoryFs>,
    key: String,
}

impl Drop for MountGuard {
    fn drop(&mut self) {
        self.fs.remove_subtree(&self.key);
    }
}

/// A single-stream (plus optional siblings) mount, exposing the primary synthetic
/// path and its own [`Hookfs`] context. Returned by [`mount_read_seek`].
#[derive(Debug)]
pub struct MountedPath {
    hookfs: Hookfs,
    primary: PathBuf,
    guards: Vec<MountGuard>,
}

impl MountedPath {
    /// The synthetic absolute path of the primary mounted stream.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.primary
    }

    /// The synthetic absolute path for a sibling `logical_name`.
    #[must_use]
    pub fn path_for(&self, logical_name: &str) -> PathBuf {
        self.hookfs.path_for(logical_name)
    }

    /// The reserved synthetic directory the mount lives in.
    #[must_use]
    pub fn virtual_dir(&self) -> &Path {
        self.hookfs.virtual_dir()
    }

    /// The mount context (to install with custom options or add more siblings).
    #[must_use]
    pub fn hookfs(&self) -> &Hookfs {
        &self.hookfs
    }

    /// Add a sibling stream (e.g. the clip's `.sidecar`).
    ///
    /// # Errors
    /// As [`Hookfs::mount`].
    pub fn with_read_seek<R: Read + Seek + Send + 'static>(
        mut self,
        logical_name: &str,
        stream: R,
    ) -> Result<Self> {
        self.guards.push(self.hookfs.mount(logical_name, stream)?);
        Ok(self)
    }

    /// Install the hooks for this mount.
    ///
    /// # Errors
    /// As [`crate::install`].
    pub fn install(&self, opts: Options) -> Result<InstallGuard> {
        self.hookfs.install(opts)
    }
}

/// Mount one owned `Read + Seek` stream under a reserved synthetic path, returning
/// a [`MountedPath`] whose [`path`](MountedPath::path) is the logical name to hand
/// to the native library.
///
/// # Errors
/// As [`Hookfs::mount`].
pub fn mount_read_seek<R: Read + Seek + Send + 'static>(
    logical_name: &str,
    stream: R,
) -> Result<MountedPath> {
    let hookfs = Hookfs::new();
    let primary = hookfs.path_for(logical_name);
    let guard = hookfs.mount(logical_name, stream)?;
    Ok(MountedPath {
        hookfs,
        primary,
        guards: vec![guard],
    })
}

/// A boxed [`FileStream`] re-wrapped so it satisfies `Read + Seek + Send`.
struct BoxedStream(Box<dyn FileStream>);

impl Read for BoxedStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl Seek for BoxedStream {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.seek(pos)
    }
}

/// Validate a logical mount name: relative, no `..`, no NUL, no drive/stream marker.
fn validate_logical(name: &str) -> std::result::Result<(), &'static str> {
    if name.is_empty() {
        return Err("empty name");
    }
    if name.contains('\0') {
        return Err("embedded NUL");
    }
    if is_absolute(name) || name.starts_with('\\') || name.starts_with('/') {
        return Err("must be a relative name");
    }
    // A joined name that would be lexically unsafe (`..`, stream markers) is rejected.
    if !is_lexically_safe(name) {
        return Err("`..` escape or stream/device marker");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn mount_exposes_reserved_path_and_lists_siblings() {
        let mounted = mount_read_seek("sample.braw", Cursor::new(vec![1u8, 2, 3]))
            .unwrap()
            .with_read_seek("sample.sidecar", Cursor::new(b"{}".to_vec()))
            .unwrap();
        let dir = mounted.virtual_dir().to_owned();
        assert!(mounted.path().starts_with(&dir));
        // The provider sees both siblings in the reserved directory.
        let listing = mounted.hookfs().vfs().read_dir(&dir).unwrap().unwrap();
        assert_eq!(listing.len(), 2);
    }

    #[test]
    fn rejects_empty_and_traversal_names() {
        let fs = Hookfs::new();
        assert!(fs.mount("", Cursor::new(vec![])).is_err());
        assert!(fs.mount("ok.braw", Cursor::new(vec![])).is_ok());
    }

    #[cfg(windows)]
    #[test]
    fn rejects_windows_unsafe_names() {
        let fs = Hookfs::new();
        assert!(fs.mount("..\\escape", Cursor::new(vec![])).is_err());
        assert!(fs.mount("C:\\abs.braw", Cursor::new(vec![])).is_err());
        assert!(fs.mount("clip.braw:stream", Cursor::new(vec![])).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_posix_unsafe_names() {
        let fs = Hookfs::new();
        assert!(fs.mount("../escape", Cursor::new(vec![])).is_err());
        assert!(fs.mount("/abs.braw", Cursor::new(vec![])).is_err());
        assert!(fs.mount("sub/../../escape", Cursor::new(vec![])).is_err());
        // A colon is a legal POSIX file-name byte.
        assert!(fs.mount("clip:1.braw", Cursor::new(vec![])).is_ok());
    }

    #[test]
    fn dropping_mount_guard_unmounts() {
        let fs = Hookfs::new();
        let guard = fs.mount("x.braw", Cursor::new(vec![0u8; 4])).unwrap();
        let path = fs.path_for("x.braw");
        assert!(fs.vfs().metadata(&path).is_some());
        drop(guard);
        assert!(fs.vfs().metadata(&path).is_none());
    }
}
