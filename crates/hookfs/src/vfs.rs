//! The virtual-filesystem core: the [`FileStream`] byte-stream trait, the
//! [`VirtualFs`] provider trait, the metadata/directory value types, and the
//! shared-source machinery that gives every native open an **independent logical
//!

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// A virtualized file: any seekable byte stream that can be handed to a native
/// consumer. [`size`](FileStream::size) answers `stat`/`GetFileSizeEx` without the
/// caller performing a seek dance.
///
/// The default `size` implementation seeks to the end and restores the position,
/// so it needs `&mut self` (a plain `&self` cannot seek). For the shared-source
/// cursor this is already O(1): its `seek(End)` reads the length cached once at
/// mount time and only moves the private cursor, so no per-call physical seek
/// happens on the hot metadata path.
pub trait FileStream: Read + Seek + Send + 'static {
    /// Logical length of the stream in bytes.
    ///
    /// The default probes it via `seek(End)` + restore.
    fn size(&mut self) -> io::Result<u64> {
        let current = self.stream_position()?;
        let end = self.seek(SeekFrom::End(0))?;
        if current != end {
            self.seek(SeekFrom::Start(current))?;
        }
        Ok(end)
    }
}

/// Every `Read + Seek + Send + 'static` value is a [`FileStream`] with the default
/// seek-based `size`. This is the blanket impl that lets callers mount an
/// in-memory `Cursor`, a decrypting reader, an HTTP-range reader, etc. without
/// implementing anything.
impl<T: Read + Seek + Send + 'static> FileStream for T {}

/// A **writable** virtual file: a [`FileStream`] that also supports write, truncate
/// [`VirtualFs::open_write`] and driven by the write shims (`WriteFile`/`write`/
/// `SetEndOfFile`/`ftruncate`/`FlushFileBuffers`/`fsync`).
///
/// Reads/seeks come from the [`FileStream`] supertrait; a write happens at the
/// current cursor and **grows** the file (a seek past EOF followed by a write
/// zero-fills the gap, mirroring OS files). [`set_len`](WritableFile::set_len)
/// truncates or extends. There is deliberately **no** blanket impl: a plain
/// `Read + Seek` reader is not writable, so a provider must return a type that
/// genuinely backs writes (the built-in [`MemoryFs`](crate::MemoryFs) is the
/// batteries-included one).
pub trait WritableFile: FileStream + Write {
    /// Set the file length, truncating or zero-extending to `len` bytes.
    fn set_len(&mut self, len: u64) -> io::Result<()>;
}

/// Synthetic volume geometry for the `statvfs`/`GetDiskFreeSpace` shims
/// capacity**; writes past `available` fail with `ENOSPC`/`ERROR_DISK_FULL`.
#[derive(Debug, Clone, Copy)]
pub struct VolumeInfo {
    /// Total capacity of the synthetic volume, in bytes.
    pub capacity: u64,
    /// Currently-available (free) space, in bytes.
    pub available: u64,
    /// Allocation block size the synthetic geometry is expressed in.
    pub block_size: u32,
}

/// The `io::Error` a full synthetic volume returns, carrying the platform's native
/// "disk full" code so the shims map it to `ENOSPC` (POSIX) / `ERROR_DISK_FULL`
/// (Win32) via each mapper's `raw_os_error` pass-through.
#[must_use]
pub(crate) fn no_space() -> io::Error {
    // `ENOSPC` is 28 on Linux, macOS and iOS; `ERROR_DISK_FULL` is 112 on Win32.
    #[cfg(windows)]
    {
        io::Error::from_raw_os_error(112)
    }
    #[cfg(not(windows))]
    {
        io::Error::from_raw_os_error(28)
    }
}

/// The `io::Error` returned when a write is attempted on a non-writable handle.
/// Used by the backend registry's [`OpenStream`](crate::registry) fallback arms.
#[cfg(hookfs_backend)]
#[must_use]
pub(crate) fn read_only() -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, "virtual file is read-only")
}

/// Metadata for a virtual path, mirroring the subset of `stat`/
/// `BY_HANDLE_FILE_INFORMATION` the shims synthesize.
#[derive(Debug, Clone)]
pub struct VfsMetadata {
    /// Logical length in bytes (ignored for directories).
    pub len: u64,
    /// Whether the entry is a directory.
    pub is_dir: bool,
    /// Whether the entry is read-only (always `true` in the read-only milestone).
    pub readonly: bool,
    /// Last-modification time, if the provider tracks one.
    pub mtime: Option<SystemTime>,
}

impl VfsMetadata {
    /// Metadata for a read-only regular file of `len` bytes.
    #[must_use]
    pub fn file(len: u64) -> Self {
        Self {
            len,
            is_dir: false,
            readonly: true,
            mtime: None,
        }
    }

    /// without the read-only attribute so a consumer that checks writability before
    /// opening for write (e.g. `FILE_ATTRIBUTE_READONLY`) proceeds.
    #[must_use]
    pub fn file_rw(len: u64) -> Self {
        Self {
            len,
            is_dir: false,
            readonly: false,
            mtime: None,
        }
    }

    /// Metadata for a directory.
    #[must_use]
    pub fn dir() -> Self {
        Self {
            len: 0,
            is_dir: true,
            readonly: true,
            mtime: None,
        }
    }
}

/// One entry in a virtual directory listing.
#[derive(Debug, Clone)]
pub struct VfsDirEntry {
    /// The entry's file name (final path component), preserving its original case.
    pub name: std::ffi::OsString,
    /// The entry's metadata.
    pub metadata: VfsMetadata,
}

/// How a native `open`/`CreateFile` requested the stream. Mirrors the access bits
/// the shims can observe so a provider can reject writes precisely.
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)] // mirrors independent OS open flags.
pub struct OpenOptions {
    /// Read access was requested (the default for clip reads).
    pub read: bool,
    /// Write access was requested (`GENERIC_WRITE` / `O_WRONLY`/`O_RDWR`).
    pub write: bool,
    /// Creation was requested if the file does not exist (`OPEN_ALWAYS` /
    /// `CREATE_ALWAYS` / `O_CREAT` / `fopen` `w`/`a`).
    pub create: bool,
    /// Creation was requested and must fail if the file already exists
    /// (`CREATE_NEW` / `O_CREAT|O_EXCL`).
    pub create_new: bool,
    /// Truncate-on-open was requested (`CREATE_ALWAYS` / `TRUNCATE_EXISTING` /
    /// `O_TRUNC` / `fopen` `w`).
    pub truncate: bool,
    /// Append mode — writes go to the end of the file (`O_APPEND` / `fopen` `a`).
    pub append: bool,
}

impl OpenOptions {
    /// A plain read-only open.
    #[must_use]
    pub fn read_only() -> Self {
        Self {
            read: true,
            ..Self::default()
        }
    }

    /// A write open that creates and truncates (the `w` / `CREATE_ALWAYS` shape).
    #[must_use]
    pub fn write_truncate() -> Self {
        Self {
            read: false,
            write: true,
            create: true,
            truncate: true,
            ..Self::default()
        }
    }

    /// Whether this open may create the file if it is missing (`OPEN_ALWAYS` /
    /// `CREATE_ALWAYS` / `CREATE_NEW` / `O_CREAT`). `truncate` alone
    /// (`TRUNCATE_EXISTING`, `O_TRUNC` without `O_CREAT`) requires an existing file.
    #[must_use]
    pub fn creates(&self) -> bool {
        self.create || self.create_new
    }
}

/// A user-implemented virtual filesystem.
///
/// Every method returns `Option<..>`: `None` means "this path is **not** handled
/// here — fall through to the real OS". For paths under the reserved prefix,
/// `None` from the routed provider is turned into a *fail-closed native error* by
///
/// Implementations must be `Send + Sync`: the SDK decodes on many threads and the
/// shims re-enter concurrently.
pub trait VirtualFs: Send + Sync + 'static {
    /// Open a stream for `path`. `None` = not handled here.
    fn open(&self, path: &Path, opts: &OpenOptions) -> Option<io::Result<Box<dyn FileStream>>>;

    /// Metadata for `path`. `None` = not handled here.
    fn metadata(&self, path: &Path) -> Option<io::Result<VfsMetadata>>;

    /// Directory listing for `path`. `None` = not handled here.
    fn read_dir(&self, path: &Path) -> Option<io::Result<Vec<VfsDirEntry>>>;

    /// Whether `path` exists. Defaults to a successful `metadata` probe.
    /// `None` = not handled here.
    fn exists(&self, path: &Path) -> Option<bool> {
        self.metadata(path).map(|r| r.is_ok())
    }

    //
    // The engine only calls these when `Options::allow_writes` is set; with writes
    // disabled the read-only fail-closed behavior is preserved. Each defaults to
    // `None` (= not handled → the shim fails closed with a native error), so a
    // read-only provider is writable-safe without implementing anything.

    /// Open a **writable** handle for `path` per `opts` (create/truncate/append/
    /// create-new). `None` = not handled here (fail closed). Only invoked on the
    /// write path when the engine permits writes.
    fn open_write(
        &self,
        path: &Path,
        opts: &OpenOptions,
    ) -> Option<io::Result<Box<dyn WritableFile>>> {
        let _ = (path, opts);
        None
    }

    /// Remove the file (or empty directory) at `path`. `None` = not handled here.
    fn remove(&self, path: &Path) -> Option<io::Result<()>> {
        let _ = path;
        None
    }

    /// Rename/move `from` to `to` within the virtual tree (the temp-then-rename
    /// finalization a writer may perform). `None` = not handled here.
    fn rename(&self, from: &Path, to: &Path) -> Option<io::Result<()>> {
        let _ = (from, to);
        None
    }

    /// Create a directory at `path`. `None` = not handled here.
    fn create_dir(&self, path: &Path) -> Option<io::Result<()>> {
        let _ = path;
        None
    }

    /// Truncate/extend the file at `path` to `len` bytes (path-based `truncate`).
    /// `None` = not handled here.
    fn set_len(&self, path: &Path, len: u64) -> Option<io::Result<()>> {
        let _ = (path, len);
        None
    }

    /// The synthetic volume geometry (capacity/free space) reported for a virtual
    /// path by `statvfs`/`GetDiskFreeSpace`. `None` = use the shim's default roomy
    /// synthetic geometry.
    fn volume_info(&self) -> Option<VolumeInfo> {
        None
    }
}

/// A physical byte source shared by every open of one mounted file, with its
/// length captured once so metadata is answered without a seek.
///
/// Physical access is serialized by the inner mutex; each open holds an
/// independent [`SourceCursor`] that seeks the shared source before every I/O.
/// This supports concurrent BRAW read jobs and repeated opens over a single
pub(crate) struct SharedSource {
    inner: Mutex<Box<dyn FileStream>>,
    size: u64,
}

impl SharedSource {
    /// Wrap `stream`, probing and caching its size up front.
    pub(crate) fn new<R: FileStream>(mut stream: R) -> io::Result<Arc<Self>> {
        let size = stream.size()?;
        // Normalize the physical cursor to a known position; every access seeks
        // explicitly, so the starting position is irrelevant, but resetting keeps
        // the source tidy for any external inspection.
        stream.seek(SeekFrom::Start(0))?;
        Ok(Arc::new(Self {
            inner: Mutex::new(Box::new(stream)),
            size,
        }))
    }

    /// The cached logical length.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// Perform a positioned read against the shared physical stream.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.seek(SeekFrom::Start(pos))?;
        guard.read(buf)
    }
}

/// An independent logical cursor over a [`SharedSource`]. Implements
/// [`FileStream`] with a cached size; its position is private to this open, so two
/// opens of the same file never disturb each other's offset.
pub(crate) struct SourceCursor {
    source: Arc<SharedSource>,
    pos: u64,
}

impl SourceCursor {
    pub(crate) fn new(source: Arc<SharedSource>) -> Self {
        Self { source, pos: 0 }
    }
}

impl Read for SourceCursor {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.source.size() {
            return Ok(0); // At/after EOF: zero bytes, not an error.
        }
        let n = self.source.read_at(self.pos, buf)?;
        self.pos = self.pos.saturating_add(n as u64);
        Ok(n)
    }
}

impl Seek for SourceCursor {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let size = self.source.size();
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
        // Seeking past EOF is legal (mirrors OS files); reads there yield 0 bytes.
        let target = u64::try_from(target)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "seek offset overflow"))?;
        self.pos = target;
        Ok(self.pos)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.pos)
    }
}

// `SourceCursor` is a `FileStream` via the blanket impl. Its `size()` is already
// O(1): the default seeks to `End`, which reads the cached `SharedSource` length
// and only mutates the private cursor — no lock and no physical I/O.
