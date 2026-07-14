//! The open-object registry: maps the opaque native handle the library holds back
//!
//! A virtual open is backed by a **real carrier** allocated through the *original*
//! syscall — a `NUL` `HANDLE` on Windows, a `/dev/null` (or `memfd`) fd on POSIX —
//! so a handle/fd that escapes to an unhooked consumer or the kernel fails
//! predictably instead of dereferencing a fabricated value. This module keys on the
//! carrier's numeric value; the carrier lifetime and the ABI live in
//! [`crate::shims`].
//!
//! Every open file is stored behind its own `Arc<Mutex<..>>`: a shim locks the
//! registry only briefly to clone the `Arc`, then does I/O under the per-object
//! lock — so two handles proceed concurrently while operations on the *same* handle
//! serialize correctly.

use crate::vfs::{FileStream, WritableFile, read_only};
use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[cfg(windows)]
use crate::vfs::VfsDirEntry;
#[cfg(unix)]
use std::collections::HashSet;

/// A synthetic, stable volume serial for every virtual file (diagnostics only,
/// Windows `BY_HANDLE_FILE_INFORMATION` / POSIX `st_dev`).
pub(crate) const VIRTUAL_VOLUME_SERIAL: u32 = 0x484F_4B46; // "HOKF"

/// The stream behind one open carrier handle / fd — either a read-only view or a
/// is readable too); only the writable variant serves write/truncate/flush. This
/// enum is the single point every shim drives, so the read shims are unchanged and
/// the write shims add exactly the `write`/`set_len`/`flush` arms.
pub(crate) enum OpenStream {
    /// A read-only cursor over a mounted source.
    Read(Box<dyn FileStream>),
    /// A writable handle (read + write + seek + truncate + flush).
    Write(Box<dyn WritableFile>),
}

impl OpenStream {
    /// Read at the current cursor.
    pub(crate) fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Read(s) => s.read(buf),
            Self::Write(s) => s.read(buf),
        }
    }

    /// Write at the current cursor (growing the file); `Err(EROFS-like)` for a
    /// read-only handle — the shims fail closed before reaching this on that path.
    pub(crate) fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Read(_) => Err(read_only()),
            Self::Write(s) => s.write(buf),
        }
    }

    pub(crate) fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Self::Read(s) => s.seek(pos),
            Self::Write(s) => s.seek(pos),
        }
    }

    /// The current cursor position.
    pub(crate) fn stream_position(&mut self) -> io::Result<u64> {
        match self {
            Self::Read(s) => s.stream_position(),
            Self::Write(s) => s.stream_position(),
        }
    }

    /// The current logical length.
    pub(crate) fn size(&mut self) -> io::Result<u64> {
        match self {
            Self::Read(s) => s.size(),
            Self::Write(s) => s.size(),
        }
    }

    /// Truncate/extend to `len` bytes; `Err` for a read-only handle.
    pub(crate) fn set_len(&mut self, len: u64) -> io::Result<()> {
        match self {
            Self::Read(_) => Err(read_only()),
            Self::Write(s) => s.set_len(len),
        }
    }

    /// Flush buffered writes (a no-op for a read-only handle).
    pub(crate) fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Read(_) => Ok(()),
            Self::Write(s) => s.flush(),
        }
    }
}

/// The virtual state behind one open carrier handle / fd.
pub(crate) struct OpenFile {
    /// The independent read/write cursor over the mounted node.
    pub(crate) stream: OpenStream,
    /// The virtual path this handle was opened with. Backs the **stable file
    /// identity** the shims synthesize (repeated opens of the same path report the
    /// same identity) and is retained for path-returning queries.
    pub(crate) path: PathBuf,
    /// Whether the open was granted write access. Drives `deny_write_fd` and the
    /// writability the metadata shims report (dropping `FILE_ATTRIBUTE_READONLY`).
    pub(crate) writable: bool,
    /// Whether the handle was opened in overlapped (async) mode — such reads are
    /// rejected with `ERROR_NOT_SUPPORTED` (R16). Windows-only.
    #[cfg(windows)]
    pub(crate) overlapped: bool,
}

/// The virtual state behind one open carrier *find* handle
/// (`FindFirstFileExW`/`FindNextFileW`). Windows-only.
#[cfg(windows)]
pub(crate) struct FindState {
    /// The remaining matched directory entries.
    pub(crate) entries: Vec<VfsDirEntry>,
    /// Index of the next entry to yield.
    pub(crate) next: usize,
}

/// The process registry of virtual file (and, on Windows, find) handles.
#[derive(Default)]
pub(crate) struct Registry {
    files: Mutex<HashMap<usize, Arc<Mutex<OpenFile>>>>,
    #[cfg(windows)]
    finds: Mutex<HashMap<usize, Arc<Mutex<FindState>>>>,
    /// Membership of the virtual `DIR*` pointers our `opendir` handed out. The
    /// enumeration state itself lives in a `Box<VirtualDir>` reached through the
    /// raw pointer; this set lets `readdir`/`closedir` distinguish our `DIR*` from a
    /// real one (POSIX).
    #[cfg(unix)]
    dirs: Mutex<HashSet<usize>>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let files = self.files.lock().map_or(0, |m| m.len());
        let mut dbg = f.debug_struct("Registry");
        dbg.field("files", &files);
        #[cfg(windows)]
        dbg.field("finds", &self.finds.lock().map_or(0, |m| m.len()));
        #[cfg(unix)]
        dbg.field("dirs", &self.dirs.lock().map_or(0, |m| m.len()));
        dbg.finish_non_exhaustive()
    }
}

impl Registry {
    /// A fresh, empty registry.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Register a virtual file behind carrier `handle`.
    pub(crate) fn insert_file(&self, handle: usize, file: OpenFile) {
        self.files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(handle, Arc::new(Mutex::new(file)));
    }

    /// The virtual file behind `handle`, if any (cloned `Arc`).
    pub(crate) fn get_file(&self, handle: usize) -> Option<Arc<Mutex<OpenFile>>> {
        self.files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&handle)
            .cloned()
    }

    /// Remove and return the virtual file behind `handle` (on close).
    pub(crate) fn remove_file(&self, handle: usize) -> Option<Arc<Mutex<OpenFile>>> {
        self.files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&handle)
    }

    /// Register a virtual find enumeration behind carrier `handle` (Windows).
    #[cfg(windows)]
    pub(crate) fn insert_find(&self, handle: usize, state: FindState) {
        self.finds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(handle, Arc::new(Mutex::new(state)));
    }

    /// The virtual find behind `handle`, if any (cloned `Arc`) (Windows).
    #[cfg(windows)]
    pub(crate) fn get_find(&self, handle: usize) -> Option<Arc<Mutex<FindState>>> {
        self.finds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&handle)
            .cloned()
    }

    /// Remove and return the virtual find behind `handle` (on `FindClose`) (Windows).
    #[cfg(windows)]
    pub(crate) fn remove_find(&self, handle: usize) -> Option<Arc<Mutex<FindState>>> {
        self.finds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&handle)
    }

    /// Record that `dir` is one of our virtual `DIR*` pointers (POSIX `opendir`).
    #[cfg(unix)]
    pub(crate) fn insert_dir(&self, dir: usize) {
        self.dirs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(dir);
    }

    /// Whether `dir` is one of our virtual `DIR*` pointers (POSIX `readdir`).
    #[cfg(unix)]
    pub(crate) fn contains_dir(&self, dir: usize) -> bool {
        self.dirs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&dir)
    }

    /// Drop membership of a virtual `DIR*` (POSIX `closedir`); returns whether it
    /// was ours.
    #[cfg(unix)]
    pub(crate) fn remove_dir(&self, dir: usize) -> bool {
        self.dirs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&dir)
    }
}
