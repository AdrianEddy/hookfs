//! [`MemoryFs`] — the batteries-included provider: a small in-memory directory
//! rename/delete of synthetic files (the sidecar the SDK saves, the `.braw` a trim
//! job writes), behind a configurable capacity so writes past it fail `ENOSPC`.
//!
//! Three node kinds live in one tree:
//! * **read-only mounts** — an external `Read + Seek` source ([`SharedSource`]),
//!   e.g. the clip bytes;
//! * **writable in-memory files** ([`MemFile`]) — a growable buffer the write shims
//!   drive, readable back through the same path;
//! * **directories** — synthesized so sibling lookups and enumeration resolve.
//!
//! Every open (read or write) produces an **independent cursor** over the shared
//! node, so concurrent handles and repeated opens keep their own positions.

use crate::namespace::{SEP, normalize_key};
use crate::vfs::{
    FileStream, OpenOptions, SharedSource, SourceCursor, VfsDirEntry, VfsMetadata, VirtualFs,
    VolumeInfo, WritableFile, no_space,
};
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

/// The default synthetic volume capacity (1 TiB) — large enough that ordinary
/// sidecar/trim writes never hit it, while [`MemoryFs::with_capacity`] pins a small
/// budget for the `ENOSPC` tests.
const DEFAULT_CAPACITY: u64 = 1 << 40;

/// The synthetic volume shared by every writable file in one [`MemoryFs`]: a fixed
/// capacity and a running count of bytes occupied by the **live** writable files
/// ([`MemFile`]s). Read-only mounted sources are external and do **not** count
/// against it.
///
/// Accounting is exactly-once and tied by RAII to a [`MemFile`]'s real lifetime, so
/// a file unlinked while a handle is still open keeps its bytes reserved until that
/// handle closes (POSIX unlink semantics):
/// * a length change reserves the delta up front — [`resize`](Self::resize)
///   all-or-nothing for a truncate/extend, [`reserve`](Self::reserve) partial for a
///   short write — failing [`no_space`] only when a growth cannot proceed;
/// * the bytes are returned in full exactly once from [`Drop for MemFile`], when the
///   file's last [`Arc`] is dropped ([`release`](Self::release)). Unlink itself never
///   releases, so `used` can never under-count a still-referenced file or leak a
///   dropped one.
///
/// `used` is always the innermost lock (taken after a [`MemFile`]'s data lock, and
/// after the [`MemoryFs`] entries lock when a drop fires under it), so the total lock
/// order `entries → data → used` stays acyclic and deadlock-free.
struct Volume {
    capacity: u64,
    used: Mutex<u64>,
}

impl Volume {
    fn new(capacity: u64) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            used: Mutex::new(0),
        })
    }

    /// Account for a writable file changing length from `old` to `new`, **all or
    /// nothing**: a growth that would exceed the capacity reserves nothing and fails
    /// with `ENOSPC`/`ERROR_DISK_FULL`; a shrink releases the difference. Used by the
    /// truncate/extend path ([`MemFile::set_len`]), which — like `ftruncate` — either
    /// resizes fully or fails, never partially.
    fn resize(&self, old: u64, new: u64) -> io::Result<()> {
        let mut used = self
            .used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if new >= old {
            let grow = new - old;
            if used.saturating_add(grow) > self.capacity {
                return Err(no_space());
            }
            *used += grow;
        } else {
            *used = used.saturating_sub(old - new);
        }
        Ok(())
    }

    /// Reserve capacity to grow a file from `old` toward `desired_new` bytes
    /// (`old <= desired_new`), granting **as much as the free capacity allows** and
    /// returning the length actually cleared to occupy (`old <= granted <= desired_new`).
    /// The `granted - old` delta is added to `used`; a caller that cannot place any
    /// data within a partial grant hands the unused bytes back via
    /// [`release`](Self::release). This is the seam that turns a write past capacity
    /// into a real short write instead of an all-or-nothing failure.
    fn reserve(&self, old: u64, desired_new: u64) -> u64 {
        let mut used = self
            .used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let want = desired_new.saturating_sub(old);
        let free = self.capacity.saturating_sub(*used);
        let grant = want.min(free);
        *used += grant;
        old + grant
    }

    /// Return `len` bytes to the volume. Invoked **only** from [`Drop for MemFile`]
    /// (the single release seam) and from the short-write rollback that hands back a
    /// speculative gap reservation it could not use — never from an unlink, which
    /// keeps the bytes reserved until the backing itself is dropped.
    fn release(&self, len: u64) {
        let mut used = self
            .used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *used = used.saturating_sub(len);
    }

    fn info(&self) -> VolumeInfo {
        let used = *self
            .used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        VolumeInfo {
            capacity: self.capacity,
            available: self.capacity.saturating_sub(used),
            block_size: 4096,
        }
    }
}

/// A writable in-memory file: a growable byte buffer plus its capacity-accounting
/// volume and modification time. Shared behind an `Arc` by every open handle, each
/// of which holds its own cursor ([`MemFileHandle`]); the buffer serializes on its
/// own mutex so concurrent writers stay consistent.
pub(crate) struct MemFile {
    data: Mutex<Vec<u8>>,
    volume: Arc<Volume>,
    mtime: Mutex<Option<SystemTime>>,
}

impl MemFile {
    fn new(volume: Arc<Volume>) -> Arc<Self> {
        Arc::new(Self {
            data: Mutex::new(Vec::new()),
            volume,
            mtime: Mutex::new(Some(SystemTime::now())),
        })
    }

    fn lock_data(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Current length in bytes.
    fn len(&self) -> u64 {
        self.lock_data().len() as u64
    }

    /// Modification time, if tracked.
    fn mtime(&self) -> Option<SystemTime> {
        *self
            .mtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn touch(&self) {
        *self
            .mtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(SystemTime::now());
    }

    /// A byte-for-byte copy of the current contents (diagnostics/tests/read-back).
    fn snapshot(&self) -> Vec<u8> {
        self.lock_data().clone()
    }

    /// Copy up to `buf.len()` bytes starting at `pos` into `buf`; `pos` at/after EOF
    /// yields zero bytes (not an error), mirroring an OS file.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> usize {
        let data = self.lock_data();
        let Ok(start) = usize::try_from(pos) else {
            return 0;
        };
        let Some(src) = data.get(start..) else {
            return 0;
        };
        let n = src.len().min(buf.len());
        let (Some(dst), Some(src)) = (buf.get_mut(..n), src.get(..n)) else {
            return 0;
        };
        dst.copy_from_slice(src);
        n
    }

    /// Write `buf` at `pos`, growing (and zero-filling any gap past the old EOF) as
    /// needed and capacity-checked. Returns the number of bytes **actually written**:
    /// `buf.len()` when it all fits, a **short count** when only part fits within the
    /// remaining capacity (mirroring `write(2)`/`WriteFile`), and `Err(ENOSPC)` only
    /// when not a single byte fits.
    fn write_at(&self, pos: u64, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0); // a zero-length write is a no-op; it never grows the file.
        }
        let mut data = self.lock_data();
        let old = data.len() as u64;
        let desired_end = pos
            .checked_add(buf.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "write offset overflow"))?;
        // The length the file would reach if the whole write fit.
        let desired_new = desired_end.max(old);
        // Reserve as much growth as the volume can spare; `granted_new` is the length
        // we are actually cleared to occupy (`old <= granted_new <= desired_new`).
        let granted_new = self.volume.reserve(old, desired_new);
        // Bytes of `buf` that land inside `[pos, granted_new)`. Zero when `pos` is at
        // or past the grant — e.g. a seek-past-EOF whose zero-fill gap alone exhausts
        // the capacity, so no byte of `buf` is reachable.
        let slots = usize::try_from(granted_new.saturating_sub(pos)).unwrap_or(usize::MAX);
        let n = slots.min(buf.len());
        if n == 0 {
            // Not one byte fit: undo the speculative gap reservation and fail closed
            // with ENOSPC, exactly as `write(2)` does when zero bytes can be written.
            self.volume.release(granted_new.saturating_sub(old));
            return Err(no_space());
        }
        // `granted_new == max(old, pos + n)`, so it is the file's exact new length. If
        // the granted length or `pos` is not addressable on this platform (a 32-bit
        // `usize` overflow), hand the speculative reservation back before failing — a
        // reservation must never outlive a write that did not happen.
        let (Ok(new_len), Ok(start)) = (usize::try_from(granted_new), usize::try_from(pos)) else {
            self.volume.release(granted_new.saturating_sub(old));
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "write offset overflow",
            ));
        };
        if data.len() < new_len {
            data.resize(new_len, 0); // zero-fill the gap past the old EOF, then extend
        }
        if let (Some(dst), Some(src)) = (data.get_mut(start..start.saturating_add(n)), buf.get(..n))
        {
            dst.copy_from_slice(src);
        }
        drop(data);
        self.touch();
        Ok(n)
    }

    /// Truncate or zero-extend the file to `len` bytes, capacity-checked.
    fn set_len(&self, len: u64) -> io::Result<()> {
        let mut data = self.lock_data();
        let old = data.len() as u64;
        self.volume.resize(old, len)?;
        let target = usize::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "length overflow"))?;
        data.resize(target, 0);
        drop(data);
        self.touch();
        Ok(())
    }
}

impl Drop for MemFile {
    /// Return this file's bytes to the volume, exactly once, when its last [`Arc`] is
    /// gone — the sole capacity-release seam. Unlink (`remove`/`remove_subtree`/a
    /// `rename` overwrite) only drops the tree's reference: a file removed while a
    /// write handle is still open therefore keeps its bytes accounted (POSIX unlink
    /// here precisely once, so `used` neither leaks the file nor under-counts it.
    fn drop(&mut self) {
        // `&mut self` proves this is the last owner, so the length needs no data lock;
        // recover a poisoned buffer rather than double-panic during unwind.
        let len = self
            .data
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        self.volume.release(len as u64);
    }
}

/// An independent cursor over a [`MemFile`]. Serves both the read path (as a
/// [`FileStream`]) and the write path (as a [`WritableFile`]); two handles over one
/// file never disturb each other's position.
pub(crate) struct MemFileHandle {
    file: Arc<MemFile>,
    pos: u64,
}

impl MemFileHandle {
    fn new(file: Arc<MemFile>) -> Self {
        Self { file, pos: 0 }
    }

    /// A handle positioned at end-of-file (append mode).
    fn at_end(file: Arc<MemFile>) -> Self {
        let pos = file.len();
        Self { file, pos }
    }
}

impl Read for MemFileHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.file.read_at(self.pos, buf);
        self.pos = self.pos.saturating_add(n as u64);
        Ok(n)
    }
}

impl Write for MemFileHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.file.write_at(self.pos, buf)?;
        self.pos = self.pos.saturating_add(n as u64);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(()) // an in-memory buffer has nothing to flush.
    }
}

impl Seek for MemFileHandle {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let size = self.file.len();
        let target: i128 = match pos {
            SeekFrom::Start(off) => i128::from(off),
            SeekFrom::End(off) => i128::from(size) + i128::from(off),
            SeekFrom::Current(off) => i128::from(self.pos) + i128::from(off),
        };
        if target < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to a negative position",
            ));
        }
        self.pos = u64::try_from(target)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "seek offset overflow"))?;
        Ok(self.pos)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.pos)
    }
}

impl WritableFile for MemFileHandle {
    fn set_len(&mut self, len: u64) -> io::Result<()> {
        self.file.set_len(len)
    }
}

/// One node in the tree: a read-only mounted source, a writable in-memory file, or
/// a synthesized directory. Each carries its original display name (preserving
/// case) for directory listings.
enum Node {
    /// A read-only mounted external source (e.g. the clip bytes).
    File {
        source: Arc<SharedSource>,
        name: OsString,
        mtime: Option<SystemTime>,
    },
    /// A writable in-memory file (the sidecar/trim output).
    MemFile { file: Arc<MemFile>, name: OsString },
    /// A synthesized directory node.
    Dir { name: OsString },
}

/// An in-memory directory tree keyed by normalized absolute path.
///
/// Mounting a file auto-creates the ancestor directory nodes, so sibling lookups
/// (`FindFirstFileExW`, `GetFileAttributesW`, the `.sidecar` probe) resolve inside
/// the VFS. Every open produces an independent cursor over the shared node.
pub struct MemoryFs {
    entries: RwLock<HashMap<String, Node>>,
    volume: Arc<Volume>,
}

impl std::fmt::Debug for MemoryFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.entries.read().map_or(0, |e| e.len());
        f.debug_struct("MemoryFs")
            .field("entries", &count)
            .field("capacity", &self.volume.capacity)
            .finish()
    }
}

impl MemoryFs {
    /// An empty tree with the default (1 TiB) writable capacity.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// An empty tree whose writable volume holds at most `capacity` bytes; writes
    #[must_use]
    pub fn with_capacity(capacity: u64) -> Arc<Self> {
        Arc::new(Self {
            entries: RwLock::new(HashMap::new()),
            volume: Volume::new(capacity),
        })
    }

    /// Mount `source` at absolute virtual `path`, creating ancestor directories.
    ///
    /// # Mount-key uniqueness
    /// The caller must guarantee `path` is unique among *live* file mounts. The
    /// safe mount layer does so by scoping every mount under a per-mount id
    /// display name — never share a key. Two file mounts sharing one key would be
    /// a correctness bug: both would hand back the same key, so dropping either
    /// [`MountGuard`](crate::mount::MountGuard) would evict the other's still-open
    /// node (last-writer-wins). The `debug_assert!` below turns that latent bug
    /// into a loud test-time failure rather than silent data corruption.
    pub(crate) fn insert_file(
        &self,
        path: &Path,
        source: Arc<SharedSource>,
        mtime: Option<SystemTime>,
    ) -> String {
        let key = normalize_key(path);
        let name = final_component(path);
        let mut map = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(
            !matches!(
                map.get(&key),
                Some(Node::File { .. } | Node::MemFile { .. })
            ),
            "mount collision: a file is already mounted at `{key}` (caller must supply unique keys)",
        );
        map.insert(
            key.clone(),
            Node::File {
                source,
                name,
                mtime,
            },
        );
        insert_ancestor_dirs(&mut map, path);
        key
    }

    /// Pre-create an empty **writable** in-memory file at absolute virtual `path`,
    /// creating ancestor directories. Returns the normalized key. Lets a caller
    /// (e.g. the mount layer) reserve a writable sidecar/output slot up front,
    /// tracked by a [`MountGuard`](crate::mount::MountGuard).
    pub(crate) fn insert_writable(&self, path: &Path) -> String {
        let key = normalize_key(path);
        let name = final_component(path);
        let file = MemFile::new(self.volume.clone());
        let mut map = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.insert(key.clone(), Node::MemFile { file, name });
        insert_ancestor_dirs(&mut map, path);
        key
    }

    /// A byte-for-byte copy of the writable file at `path`, or `None` if `path` is
    /// not a writable in-memory file (diagnostics / read-back of written content).
    #[must_use]
    pub fn snapshot(&self, path: &Path) -> Option<Vec<u8>> {
        let key = normalize_key(path);
        let map = self
            .entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match map.get(&key)? {
            Node::MemFile { file, .. } => Some(file.snapshot()),
            _ => None,
        }
    }

    /// Remove every entry whose key is `key` or a descendant of it (used by
    /// [`MountGuard`](crate::mount::MountGuard) to drop a whole mount subtree),
    /// then prune any now-childless ancestor directories that were auto-created
    /// for the mount.
    ///
    /// Pruning walks up from `key` and removes each ancestor `Dir` node that has
    /// no remaining children, stopping at the first ancestor still in use. This
    /// keeps a clip's teardown to *its own* subtree — including its per-mount id
    /// directory — without disturbing sibling clips that still hold entries under
    /// a shared ancestor (a childless synthetic directory carries no data and is
    /// re-created on demand by the next mount). A writable file inside the subtree
    /// returns its capacity through [`Drop for MemFile`] when its last handle closes,
    /// so an entry unlinked here while still open stays accounted until then.
    pub(crate) fn remove_subtree(&self, key: &str) {
        let mut map = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Dropping each in-subtree node releases a `MemFile`'s capacity via its `Drop`
        // once no handle references it (RAII), so no explicit release is needed here.
        map.retain(|k, _| !(k == key || k.strip_prefix(key).is_some_and(|r| r.starts_with(SEP))));
        let mut ancestor = parent_key(key).map(str::to_owned);
        while let Some(dir) = ancestor {
            // Only prune a synthesized directory node that is now childless.
            if !matches!(map.get(&dir), Some(Node::Dir { .. })) {
                break;
            }
            if map.keys().any(|k| parent_key(k) == Some(dir.as_str())) {
                break;
            }
            let next = parent_key(&dir).map(str::to_owned);
            map.remove(&dir);
            ancestor = next;
        }
    }

    /// Number of live entries (diagnostics/tests).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().map_or(0, |e| e.len())
    }

    /// Whether the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn metadata_for(map: &HashMap<String, Node>, key: &str) -> Option<VfsMetadata> {
        match map.get(key)? {
            Node::File { source, mtime, .. } => {
                let mut meta = VfsMetadata::file(source.size());
                meta.mtime = *mtime;
                Some(meta)
            }
            Node::MemFile { file, .. } => {
                let mut meta = VfsMetadata::file_rw(file.len());
                meta.mtime = file.mtime();
                Some(meta)
            }
            Node::Dir { .. } => Some(VfsMetadata::dir()),
        }
    }
}

impl VirtualFs for MemoryFs {
    fn open(&self, path: &Path, _opts: &OpenOptions) -> Option<io::Result<Box<dyn FileStream>>> {
        let key = normalize_key(path);
        let map = self
            .entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match map.get(&key)? {
            // A read view over the mounted source: each open owns its own cursor.
            Node::File { source, .. } => Some(Ok(Box::new(SourceCursor::new(source.clone())))),
            // A read view over the writable file, reflecting its current contents.
            Node::MemFile { file, .. } => Some(Ok(Box::new(MemFileHandle::new(file.clone())))),
            Node::Dir { .. } => Some(Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "path is a directory",
            ))),
        }
    }

    fn open_write(
        &self,
        path: &Path,
        opts: &OpenOptions,
    ) -> Option<io::Result<Box<dyn WritableFile>>> {
        let key = normalize_key(path);
        let mut map = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let file = match map.get(&key) {
            Some(Node::Dir { .. }) => {
                return Some(Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "path is a directory",
                )));
            }
            // A read-only mounted source cannot be opened for writing.
            Some(Node::File { .. }) => {
                return Some(Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "virtual file is read-only",
                )));
            }
            Some(Node::MemFile { file, .. }) => {
                if opts.create_new {
                    return Some(Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "file already exists",
                    )));
                }
                file.clone()
            }
            None => {
                if !opts.creates() {
                    // Write-open of a missing file without a create disposition.
                    return Some(Err(io::Error::from(io::ErrorKind::NotFound)));
                }
                let file = MemFile::new(self.volume.clone());
                let name = final_component(path);
                map.insert(
                    key,
                    Node::MemFile {
                        file: file.clone(),
                        name,
                    },
                );
                insert_ancestor_dirs(&mut map, path);
                file
            }
        };
        drop(map); // do I/O without holding the tree lock.
        if opts.truncate
            && let Err(err) = file.set_len(0)
        {
            return Some(Err(err));
        }
        let handle = if opts.append {
            MemFileHandle::at_end(file)
        } else {
            MemFileHandle::new(file)
        };
        Some(Ok(Box::new(handle)))
    }

    fn metadata(&self, path: &Path) -> Option<io::Result<VfsMetadata>> {
        let key = normalize_key(path);
        let map = self
            .entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::metadata_for(&map, &key).map(Ok)
    }

    fn read_dir(&self, path: &Path) -> Option<io::Result<Vec<VfsDirEntry>>> {
        let dir_key = normalize_key(path);
        let map = self
            .entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Only answer for a known directory node.
        if !matches!(map.get(&dir_key), Some(Node::Dir { .. })) {
            return None;
        }
        let mut out = Vec::new();
        for (key, node) in map.iter() {
            if parent_key(key) == Some(dir_key.as_str()) {
                let (name, metadata) = match node {
                    Node::File {
                        name,
                        source,
                        mtime,
                    } => {
                        let mut meta = VfsMetadata::file(source.size());
                        meta.mtime = *mtime;
                        (name.clone(), meta)
                    }
                    Node::MemFile { name, file } => {
                        let mut meta = VfsMetadata::file_rw(file.len());
                        meta.mtime = file.mtime();
                        (name.clone(), meta)
                    }
                    Node::Dir { name } => (name.clone(), VfsMetadata::dir()),
                };
                out.push(VfsDirEntry { name, metadata });
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Some(Ok(out))
    }

    fn remove(&self, path: &Path) -> Option<io::Result<()>> {
        let key = normalize_key(path);
        let mut map = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match map.get(&key) {
            None => Some(Err(io::Error::from(io::ErrorKind::NotFound))),
            Some(Node::Dir { .. }) => {
                // Only an empty directory can be removed.
                if map.keys().any(|k| parent_key(k) == Some(key.as_str())) {
                    return Some(Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "directory not empty",
                    )));
                }
                map.remove(&key);
                Some(Ok(()))
            }
            // Unlink the entry; a `MemFile`'s capacity is released by its `Drop` once
            // the last handle closes (POSIX unlink semantics), not here.
            Some(Node::MemFile { .. } | Node::File { .. }) => {
                map.remove(&key);
                Some(Ok(()))
            }
        }
    }

    /// Rename/move `from` onto `to` with `rename(2)` / `MoveFileExW`
    /// (`MOVEFILE_REPLACE_EXISTING`) semantics:
    /// * `from` missing → `NotFound`; `to` equal to `from` → a no-op;
    /// * an existing destination **file** is atomically replaced (its bytes released
    ///   through [`Drop for MemFile`]);
    /// * an existing **empty** destination directory is replaced; a **non-empty** one
    ///   fails `DirectoryNotEmpty` (no silent merge of unrelated children);
    /// * a file→directory or directory→file type mismatch fails `IsADirectory` /
    ///   `NotADirectory`; making a directory a subdirectory of itself fails
    ///   `InvalidInput` (`EINVAL`).
    ///
    /// Only after the destination is validated and any single replaced node unlinked
    /// does `from`'s whole subtree move onto the `to` prefix, so a destination is
    /// never left holding stray children.
    fn rename(&self, from: &Path, to: &Path) -> Option<io::Result<()>> {
        let from_key = normalize_key(from);
        let to_key = normalize_key(to);
        let mut map = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(from_node) = map.get(&from_key) else {
            return Some(Err(io::Error::from(io::ErrorKind::NotFound)));
        };
        if to_key == from_key {
            return Some(Ok(())); // renaming onto itself is a no-op.
        }
        let from_is_dir = matches!(from_node, Node::Dir { .. });
        // EINVAL: a directory cannot be moved into its own subtree (`to` under `from`).
        if from_is_dir
            && to_key
                .strip_prefix(&from_key)
                .is_some_and(|r| r.starts_with(SEP))
        {
            return Some(Err(io::Error::from(io::ErrorKind::InvalidInput)));
        }
        // Validate the destination per the replace/merge rules, and unlink the single
        // node it may replace (a file, or an empty directory) before the move.
        if let Some(dest) = map.get(&to_key) {
            let dest_is_dir = matches!(dest, Node::Dir { .. });
            match (from_is_dir, dest_is_dir) {
                (false, true) => return Some(Err(io::Error::from(io::ErrorKind::IsADirectory))),
                (true, false) => return Some(Err(io::Error::from(io::ErrorKind::NotADirectory))),
                (true, true) => {
                    // Both directories: the destination must be empty to be replaced.
                    if map.keys().any(|k| parent_key(k) == Some(to_key.as_str())) {
                        return Some(Err(io::Error::from(io::ErrorKind::DirectoryNotEmpty)));
                    }
                    map.remove(&to_key);
                }
                // Both files: replace — dropping the old node releases its capacity.
                (false, false) => {
                    map.remove(&to_key);
                }
            }
        }
        // Move `from` and its whole subtree onto the `to` prefix (covers a temp-then-
        // rename finalization of a single file *and* a directory move). The
        // destination prefix is now clear, so no insert overwrites a stray node.
        let to_move: Vec<String> = map
            .keys()
            .filter(|k| {
                *k == &from_key
                    || k.strip_prefix(&from_key)
                        .is_some_and(|r| r.starts_with(SEP))
            })
            .cloned()
            .collect();
        for k in to_move {
            if let Some(mut node) = map.remove(&k) {
                let new_key = rekey(&k, &from_key, &to_key);
                if k == from_key {
                    rename_node(&mut node, final_component(to));
                }
                map.insert(new_key, node);
            }
        }
        insert_ancestor_dirs(&mut map, to);
        Some(Ok(()))
    }

    fn create_dir(&self, path: &Path) -> Option<io::Result<()>> {
        let key = normalize_key(path);
        let mut map = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.contains_key(&key) {
            return Some(Err(io::Error::from(io::ErrorKind::AlreadyExists)));
        }
        let name = final_component(path);
        map.insert(key, Node::Dir { name });
        insert_ancestor_dirs(&mut map, path);
        Some(Ok(()))
    }

    fn set_len(&self, path: &Path, len: u64) -> Option<io::Result<()>> {
        let key = normalize_key(path);
        let map = self
            .entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match map.get(&key) {
            Some(Node::MemFile { file, .. }) => Some(file.set_len(len)),
            Some(_) => Some(Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "virtual file is read-only",
            ))),
            None => Some(Err(io::Error::from(io::ErrorKind::NotFound))),
        }
    }

    fn volume_info(&self) -> Option<VolumeInfo> {
        Some(self.volume.info())
    }
}

/// Re-target key `k` from the `from` prefix onto the `to` prefix (for a subtree
/// move). `k` is `from` itself or a descendant of it.
fn rekey(k: &str, from: &str, to: &str) -> String {
    if k == from {
        to.to_owned()
    } else {
        // `k` starts with `from` + separator; keep the suffix after `from`.
        format!("{to}{}", &k[from.len()..])
    }
}

/// Replace a node's display name (used when a rename changes the final component).
fn rename_node(node: &mut Node, name: OsString) {
    match node {
        Node::File { name: n, .. } | Node::MemFile { name: n, .. } | Node::Dir { name: n } => {
            *n = name;
        }
    }
}

/// Insert `Dir` nodes for every ancestor directory of `path` that is missing.
fn insert_ancestor_dirs(map: &mut HashMap<String, Node>, path: &Path) {
    let mut current = path.parent();
    while let Some(dir) = current {
        let key = normalize_key(dir);
        if key.is_empty() {
            break;
        }
        let name = final_component(dir);
        map.entry(key).or_insert(Node::Dir { name });
        // Stop once we reach a filesystem root (`c:\` on Windows, `/` on POSIX):
        // `Path::parent` returns `None` there on both platforms.
        if dir.parent().is_none() {
            break;
        }
        current = dir.parent();
    }
}

/// The final path component as an `OsString`, or the whole path if it has none.
fn final_component(path: &Path) -> OsString {
    path.file_name()
        .map_or_else(|| path.as_os_str().to_owned(), std::ffi::OsStr::to_owned)
}

/// Length of the filesystem-root prefix a normalized key starts with: `3` for the
/// Windows drive root `c:\` (`c`, `:`, `\`) and `1` for the POSIX root `/`. A last
/// separator that falls inside this prefix belongs to the root, whose trailing
/// separator [`parent_key`] keeps.
#[cfg(windows)]
const ROOT_LEN: usize = 3;
#[cfg(unix)]
const ROOT_LEN: usize = 1;

/// The parent key of a normalized key (everything before the last separator), or
/// `None` for a key with no separator. A filesystem root keeps its trailing
/// separator: on Windows the parent of `c:\foo` is `c:\`; on POSIX the parent of
/// `/foo` is `/`. Keys use the native [`SEP`], so this is separator-correct on both.
fn parent_key(key: &str) -> Option<&str> {
    let idx = key.rfind(SEP)?;
    if idx < ROOT_LEN {
        key.get(..=idx)
    } else {
        key.get(..idx)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read, Seek, SeekFrom, Write};
    use std::path::PathBuf;

    /// The reserved prefix in the form `normalize_key` produces on this OS
    /// (`C:\__hookfs__` on Windows, `/__hookfs__` on POSIX). Tests build paths
    /// under it via [`vpath`] instead of hardcoding platform-specific literals.
    #[cfg(windows)]
    const RESERVED_PREFIX: &str = "C:\\__hookfs__";
    #[cfg(unix)]
    const RESERVED_PREFIX: &str = "/__hookfs__";

    /// A reserved path built from `parts`, joined with the native separator so it
    /// matches the keys the provider derives on this OS.
    fn vpath(parts: &[&str]) -> PathBuf {
        let mut p = PathBuf::from(RESERVED_PREFIX);
        for part in parts {
            p.push(part);
        }
        p
    }

    fn mount(fs: &MemoryFs, path: &Path, data: &[u8]) -> String {
        let src = SharedSource::new(Cursor::new(data.to_vec())).unwrap();
        fs.insert_file(path, src, None)
    }

    fn wt() -> OpenOptions {
        OpenOptions::write_truncate()
    }

    /// Extract the error from a write-open that must fail (a `Box<dyn WritableFile>`
    /// is not `Debug`, so `unwrap_err` is unavailable on the `Ok` arm).
    fn write_err(r: Option<io::Result<Box<dyn WritableFile>>>) -> io::Error {
        match r {
            Some(Err(e)) => e,
            _ => panic!("expected the write-open to fail"),
        }
    }

    #[test]
    fn open_metadata_and_readdir() {
        let fs = MemoryFs::new();
        mount(&fs, &vpath(&["id", "sample.braw"]), b"hello world");
        mount(&fs, &vpath(&["id", "sample.sidecar"]), b"{}");

        let meta = fs
            .metadata(&vpath(&["id", "sample.braw"]))
            .unwrap()
            .unwrap();
        assert_eq!(meta.len, 11);
        assert!(!meta.is_dir);

        let dir = fs.metadata(&vpath(&["id"])).unwrap().unwrap();
        assert!(dir.is_dir);

        // Sibling grouping: both files list under their shared parent directory.
        let entries = fs.read_dir(&vpath(&["id"])).unwrap().unwrap();
        let names: Vec<_> = entries
            .iter()
            .map(|e| e.name.to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"sample.braw".to_string()));
        assert!(names.contains(&"sample.sidecar".to_string()));
    }

    #[test]
    fn independent_cursors_over_shared_source() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "a.bin"]);
        mount(&fs, &p, b"0123456789");

        let mut a = fs.open(&p, &OpenOptions::read_only()).unwrap().unwrap();
        let mut b = fs.open(&p, &OpenOptions::read_only()).unwrap().unwrap();

        a.seek(SeekFrom::Start(5)).unwrap();
        let mut ba = [0u8; 3];
        b.read_exact(&mut ba).unwrap();
        assert_eq!(&ba, b"012"); // b unaffected by a's seek
        let mut aa = [0u8; 3];
        a.read_exact(&mut aa).unwrap();
        assert_eq!(&aa, b"567");
    }

    #[test]
    fn create_write_read_back() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "out.sidecar"]);
        // Not present until created.
        assert!(fs.metadata(&p).is_none());

        let mut w = fs.open_write(&p, &wt()).unwrap().unwrap();
        w.write_all(b"hello ").unwrap();
        w.write_all(b"world").unwrap();
        drop(w);

        // Metadata reflects the written length and reports writable (not read-only).
        let meta = fs.metadata(&p).unwrap().unwrap();
        assert_eq!(meta.len, 11);
        assert!(!meta.readonly);

        // Readable back through the same path, and via the direct snapshot.
        let mut r = fs.open(&p, &OpenOptions::read_only()).unwrap().unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello world");
        assert_eq!(fs.snapshot(&p).unwrap(), b"hello world");
    }

    #[test]
    fn seek_past_eof_zero_fills_then_writes() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "sparse.bin"]);
        let mut w = fs.open_write(&p, &wt()).unwrap().unwrap();
        w.seek(SeekFrom::Start(4)).unwrap();
        w.write_all(b"XY").unwrap();
        drop(w);
        assert_eq!(fs.snapshot(&p).unwrap(), b"\0\0\0\0XY");
    }

    #[test]
    fn truncate_grows_and_shrinks() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "t.bin"]);
        let mut w = fs.open_write(&p, &wt()).unwrap().unwrap();
        w.write_all(b"abcdef").unwrap();
        w.set_len(3).unwrap();
        assert_eq!(fs.snapshot(&p).unwrap(), b"abc");
        w.set_len(5).unwrap(); // zero-extend
        assert_eq!(fs.snapshot(&p).unwrap(), b"abc\0\0");
    }

    #[test]
    fn append_positions_at_end() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "log.txt"]);
        fs.open_write(&p, &wt())
            .unwrap()
            .unwrap()
            .write_all(b"first")
            .unwrap();
        let mut a = fs
            .open_write(
                &p,
                &OpenOptions {
                    write: true,
                    create: true,
                    append: true,
                    ..OpenOptions::default()
                },
            )
            .unwrap()
            .unwrap();
        a.write_all(b"+second").unwrap();
        assert_eq!(fs.snapshot(&p).unwrap(), b"first+second");
    }

    #[test]
    fn independent_write_cursors_and_shared_content() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "shared.bin"]);
        // Pre-size the file so both handles address the same region.
        fs.open_write(&p, &wt())
            .unwrap()
            .unwrap()
            .set_len(8)
            .unwrap();

        let mut a = fs
            .open_write(
                &p,
                &OpenOptions {
                    write: true,
                    ..OpenOptions::default()
                },
            )
            .unwrap()
            .unwrap();
        let mut b = fs
            .open_write(
                &p,
                &OpenOptions {
                    write: true,
                    ..OpenOptions::default()
                },
            )
            .unwrap()
            .unwrap();
        a.seek(SeekFrom::Start(0)).unwrap();
        b.seek(SeekFrom::Start(4)).unwrap();
        a.write_all(b"AAAA").unwrap();
        b.write_all(b"BBBB").unwrap();
        assert_eq!(fs.snapshot(&p).unwrap(), b"AAAABBBB");
    }

    #[test]
    fn capacity_enforced_with_enospc() {
        let fs = MemoryFs::with_capacity(8);
        let p = vpath(&["id", "big.bin"]);
        let mut w = fs.open_write(&p, &wt()).unwrap().unwrap();
        w.write_all(b"12345678").unwrap(); // exactly fills the 8-byte volume
        let over = w.write_all(b"9");
        let err = over.unwrap_err();
        // ENOSPC (28) on POSIX, ERROR_DISK_FULL (112) on Win32.
        #[cfg(unix)]
        assert_eq!(err.raw_os_error(), Some(28));
        #[cfg(windows)]
        assert_eq!(err.raw_os_error(), Some(112));

        // Freeing the file returns the space to the volume.
        drop(w);
        fs.remove(&p).unwrap().unwrap();
        assert_eq!(fs.volume_info().unwrap().available, 8);
    }

    #[test]
    fn write_past_capacity_is_a_short_write() {
        // Real `write(2)` semantics: a write that only partially fits writes what it
        // can and returns the short count; ENOSPC is reserved for when *nothing* fits.
        let fs = MemoryFs::with_capacity(8);
        let p = vpath(&["id", "short.bin"]);
        let mut w = fs.open_write(&p, &wt()).unwrap().unwrap();
        w.write_all(b"123456").unwrap(); // 6 bytes; 2 of the 8 remain

        // Only the 2 bytes that fit are written — a short write, not an all-or-nothing
        // ENOSPC that would have written 0.
        let n = w.write(b"789AB").unwrap();
        assert_eq!(n, 2, "only the bytes that fit are written");
        assert_eq!(fs.snapshot(&p).unwrap(), b"12345678");
        assert_eq!(
            fs.volume_info().unwrap().available,
            0,
            "the volume is now exactly full"
        );

        // With zero capacity left, the next non-empty write reports ENOSPC.
        let err = w.write(b"C").unwrap_err();
        #[cfg(unix)]
        assert_eq!(err.raw_os_error(), Some(28));
        #[cfg(windows)]
        assert_eq!(err.raw_os_error(), Some(112));

        // A zero-length write is always a successful no-op, even on a full volume.
        assert_eq!(w.write(b"").unwrap(), 0);
    }

    #[test]
    fn remove_while_write_handle_open_releases_capacity_exactly_once() {
        // RAII capacity accounting: unlinking a file that still has an open write
        // handle keeps its bytes reserved (POSIX unlink semantics) and releases them
        // exactly once when the last handle drops — no leak, no under-count.
        let fs = MemoryFs::with_capacity(64);
        let p = vpath(&["id", "open.bin"]);
        let mut w = fs.open_write(&p, &wt()).unwrap().unwrap();
        w.write_all(b"0123456789").unwrap(); // 10 bytes
        assert_eq!(fs.volume_info().unwrap().available, 64 - 10);

        // Unlink while the handle is still open.
        fs.remove(&p).unwrap().unwrap();
        assert!(
            fs.metadata(&p).is_none(),
            "the unlinked path is gone from the tree"
        );
        assert_eq!(
            fs.volume_info().unwrap().available,
            64 - 10,
            "capacity stays reserved while a handle keeps the backing alive",
        );

        // Keep writing through the still-open handle; the growth is tracked against the
        // live backing (the classic leak scenario: nothing else would ever release it).
        w.write_all(b"ABCDEF").unwrap(); // now 16 bytes
        assert_eq!(fs.volume_info().unwrap().available, 64 - 16);

        // Dropping the last handle returns exactly the current byte count.
        drop(w);
        assert_eq!(
            fs.volume_info().unwrap().available,
            64,
            "capacity is returned in full, exactly once, on the final drop",
        );
    }

    #[test]
    fn create_new_fails_if_exists() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "x.bin"]);
        fs.open_write(&p, &wt()).unwrap().unwrap();
        let opts = OpenOptions {
            write: true,
            create_new: true,
            ..OpenOptions::default()
        };
        assert_eq!(
            write_err(fs.open_write(&p, &opts)).kind(),
            io::ErrorKind::AlreadyExists
        );
    }

    #[test]
    fn write_open_missing_without_create_is_enoent() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "missing.bin"]);
        let opts = OpenOptions {
            write: true,
            ..OpenOptions::default()
        };
        assert_eq!(
            write_err(fs.open_write(&p, &opts)).kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    fn read_only_mount_cannot_be_written() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "ro.braw"]);
        mount(&fs, &p, b"payload");
        let opts = OpenOptions {
            write: true,
            ..OpenOptions::default()
        };
        assert_eq!(
            write_err(fs.open_write(&p, &opts)).kind(),
            io::ErrorKind::PermissionDenied
        );
    }

    #[test]
    fn rename_moves_file_and_frees_destination() {
        let fs = MemoryFs::new();
        let tmp = vpath(&["id", "out.tmp"]);
        let fin = vpath(&["id", "out.braw"]);
        fs.open_write(&tmp, &wt())
            .unwrap()
            .unwrap()
            .write_all(b"final-bytes")
            .unwrap();
        fs.rename(&tmp, &fin).unwrap().unwrap();
        assert!(fs.metadata(&tmp).is_none());
        assert_eq!(fs.snapshot(&fin).unwrap(), b"final-bytes");
        // The renamed entry lists under its directory with the new display name.
        let names: Vec<_> = fs
            .read_dir(&vpath(&["id"]))
            .unwrap()
            .unwrap()
            .iter()
            .map(|e| e.name.to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"out.braw".to_string()));
        assert!(!names.contains(&"out.tmp".to_string()));
    }

    #[test]
    fn rename_replaces_destination_and_releases_its_capacity() {
        // POSIX/`MoveFileEx(REPLACE_EXISTING)`: renaming onto an existing file replaces
        // it, and the replaced file's capacity is released (via its `Drop`).
        let fs = MemoryFs::with_capacity(64);
        let src = vpath(&["id", "src.bin"]);
        let dst = vpath(&["id", "dst.bin"]);
        fs.open_write(&src, &wt())
            .unwrap()
            .unwrap()
            .write_all(b"NEW")
            .unwrap(); // 3 bytes
        fs.open_write(&dst, &wt())
            .unwrap()
            .unwrap()
            .write_all(b"OLD-CONTENT")
            .unwrap(); // 11 bytes
        assert_eq!(fs.volume_info().unwrap().available, 64 - 3 - 11);

        fs.rename(&src, &dst).unwrap().unwrap();
        assert!(
            fs.metadata(&src).is_none(),
            "the source is gone after the move"
        );
        assert_eq!(
            fs.snapshot(&dst).unwrap(),
            b"NEW",
            "the destination holds the source bytes"
        );
        // Only the 3 source bytes remain accounted; the replaced 11-byte file was freed.
        assert_eq!(fs.volume_info().unwrap().available, 64 - 3);
    }

    #[test]
    fn rename_onto_nonempty_directory_is_enotempty() {
        let fs = MemoryFs::new();
        // A source directory with a child.
        let src_dir = vpath(&["id", "srcdir"]);
        fs.create_dir(&src_dir).unwrap().unwrap();
        fs.open_write(&vpath(&["id", "srcdir", "a.bin"]), &wt())
            .unwrap()
            .unwrap()
            .write_all(b"a")
            .unwrap();
        // A NON-EMPTY destination directory (an unrelated child lives under it).
        let dst_dir = vpath(&["id", "dstdir"]);
        fs.create_dir(&dst_dir).unwrap().unwrap();
        fs.open_write(&vpath(&["id", "dstdir", "keep.bin"]), &wt())
            .unwrap()
            .unwrap()
            .write_all(b"keep")
            .unwrap();

        // POSIX: renaming a directory onto a non-empty directory fails ENOTEMPTY — no
        // silent merge — leaving both trees untouched.
        let err = fs.rename(&src_dir, &dst_dir).unwrap().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::DirectoryNotEmpty);
        assert_eq!(
            fs.snapshot(&vpath(&["id", "srcdir", "a.bin"])).unwrap(),
            b"a"
        );
        assert_eq!(
            fs.snapshot(&vpath(&["id", "dstdir", "keep.bin"])).unwrap(),
            b"keep",
            "the destination's unrelated child is not dropped",
        );
    }

    #[test]
    fn rename_onto_empty_directory_replaces_it() {
        let fs = MemoryFs::new();
        let src_dir = vpath(&["id", "srcdir"]);
        fs.create_dir(&src_dir).unwrap().unwrap();
        fs.open_write(&vpath(&["id", "srcdir", "a.bin"]), &wt())
            .unwrap()
            .unwrap()
            .write_all(b"a")
            .unwrap();
        // An EMPTY destination directory is replaced by the source directory.
        let dst_dir = vpath(&["id", "dstdir"]);
        fs.create_dir(&dst_dir).unwrap().unwrap();

        fs.rename(&src_dir, &dst_dir).unwrap().unwrap();
        assert!(
            fs.metadata(&src_dir).is_none(),
            "the source directory moved away"
        );
        assert_eq!(
            fs.snapshot(&vpath(&["id", "dstdir", "a.bin"])).unwrap(),
            b"a",
            "the source's child now lives under the destination path",
        );
    }

    #[test]
    fn create_dir_and_remove() {
        let fs = MemoryFs::new();
        let d = vpath(&["id", "sub"]);
        fs.create_dir(&d).unwrap().unwrap();
        assert!(fs.metadata(&d).unwrap().unwrap().is_dir);
        // Creating it again fails.
        assert_eq!(
            fs.create_dir(&d).unwrap().unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        // A non-empty directory cannot be removed.
        fs.open_write(&vpath(&["id", "sub", "f.bin"]), &wt())
            .unwrap()
            .unwrap();
        assert!(fs.remove(&d).unwrap().is_err());
    }

    // case and normalizes separators; POSIX is byte-preserving and case-sensitive.
    #[cfg(windows)]
    #[test]
    fn lookup_is_case_insensitive_and_separator_normalized() {
        let fs = MemoryFs::new();
        mount(&fs, &vpath(&["id", "Sample.Braw"]), b"x");
        // A differently-cased, forward-slashed spelling still hits.
        assert!(
            fs.metadata(Path::new("c:/__HOOKFS__/id/SAMPLE.BRAW"))
                .is_some()
        );
        // A genuinely different name misses.
        assert!(fs.metadata(&vpath(&["id", "missing"])).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn lookup_is_case_sensitive_and_separator_collapsed() {
        let fs = MemoryFs::new();
        mount(&fs, &vpath(&["id", "Sample.Braw"]), b"x");
        // Exact spelling hits; a collapsed run of separators still hits.
        assert!(
            fs.metadata(Path::new("/__hookfs__/id/Sample.Braw"))
                .is_some()
        );
        assert!(
            fs.metadata(Path::new("/__hookfs__//id//Sample.Braw"))
                .is_some()
        );
        // A differently-cased spelling misses (case-sensitive, unlike Windows).
        assert!(
            fs.metadata(Path::new("/__hookfs__/id/sample.braw"))
                .is_none()
        );
        // A genuinely different name misses.
        assert!(fs.metadata(&vpath(&["id", "missing"])).is_none());
    }

    #[test]
    fn remove_subtree_drops_entries_but_open_cursor_survives() {
        let fs = MemoryFs::new();
        let p = vpath(&["id", "a.bin"]);
        let key = mount(&fs, &p, b"payload");
        let mut open = fs.open(&p, &OpenOptions::read_only()).unwrap().unwrap();
        fs.remove_subtree(&key);
        assert!(fs.metadata(&p).is_none());
        // The already-open cursor still reads its bytes.
        let mut buf = Vec::new();
        open.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"payload");
    }

    #[test]
    fn same_name_under_distinct_mount_dirs_are_independent() {
        // Two clips share a `sample.braw` display name but live under distinct
        // per-mount id sub-directories (as the safe mount layer arranges).
        let fs = MemoryFs::new();
        let a = mount(&fs, &vpath(&["inst", "clip-a", "sample.braw"]), b"AAAA");
        let b = mount(&fs, &vpath(&["inst", "clip-b", "sample.braw"]), b"BBBB");
        assert_ne!(a, b, "distinct sub-dirs must yield distinct keys");

        // Dropping clip A's subtree must leave clip B fully intact.
        fs.remove_subtree(&a);
        assert!(
            fs.metadata(&vpath(&["inst", "clip-a", "sample.braw"]))
                .is_none()
        );
        let bmeta = fs
            .metadata(&vpath(&["inst", "clip-b", "sample.braw"]))
            .unwrap()
            .unwrap();
        assert_eq!(bmeta.len, 4);
        // Clip B still reads its OWN bytes (not clip A's).
        let mut open = fs
            .open(
                &vpath(&["inst", "clip-b", "sample.braw"]),
                &OpenOptions::read_only(),
            )
            .unwrap()
            .unwrap();
        let mut buf = Vec::new();
        open.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"BBBB");
    }

    #[test]
    fn remove_subtree_prunes_empty_ancestor_dirs_but_keeps_shared_ones() {
        let fs = MemoryFs::new();
        let a = mount(&fs, &vpath(&["inst", "clip-a", "clip.braw"]), b"x");
        let _b = mount(&fs, &vpath(&["inst", "clip-b", "clip.braw"]), b"y");
        // Ancestor dirs exist while mounted.
        assert!(
            fs.metadata(&vpath(&["inst", "clip-a"]))
                .unwrap()
                .unwrap()
                .is_dir
        );

        fs.remove_subtree(&a);
        // Clip A's own per-mount dir is pruned…
        assert!(fs.metadata(&vpath(&["inst", "clip-a"])).is_none());
        // …but the shared ancestor survives because clip B still lives under it.
        assert!(fs.metadata(&vpath(&["inst"])).unwrap().unwrap().is_dir);
        assert!(
            fs.metadata(&vpath(&["inst", "clip-b", "clip.braw"]))
                .is_some()
        );
    }

    #[test]
    fn remove_subtree_of_last_mount_prunes_up_to_root() {
        let fs = MemoryFs::new();
        let key = mount(&fs, &vpath(&["inst", "clip-a", "clip.braw"]), b"x");
        fs.remove_subtree(&key);
        // No orphaned per-clip / reserved directory nodes remain after the last
        // mount is dropped…
        assert!(fs.metadata(&vpath(&["inst", "clip-a"])).is_none());
        assert!(fs.metadata(&vpath(&["inst"])).is_none());
        assert!(fs.metadata(&vpath(&[])).is_none());
        // …only the shared filesystem-root node may persist (it is not clip state
        // and is re-used, so it is not a per-clip leak).
        assert!(
            fs.len() <= 1,
            "no per-clip directory nodes must leak (root aside), got {}",
            fs.len()
        );
    }
}
