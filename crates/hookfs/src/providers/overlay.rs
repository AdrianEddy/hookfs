//! [`OverlayFs`] — compose several [`VirtualFs`] layers into one, first-match
//! wins. Lets a caller stack, say, an in-memory clip over a read-through cache or
//! mine — ask the next layer"; the overlay only returns `None` if *every* layer
//! declines.

use crate::vfs::{
    FileStream, OpenOptions, VfsDirEntry, VfsMetadata, VirtualFs, VolumeInfo, WritableFile,
};
use std::io;
use std::path::Path;
use std::sync::Arc;

/// A stack of virtual filesystems consulted in order.
pub struct OverlayFs {
    layers: Vec<Arc<dyn VirtualFs>>,
}

impl std::fmt::Debug for OverlayFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayFs")
            .field("layers", &self.layers.len())
            .finish()
    }
}

impl OverlayFs {
    /// An overlay of `layers`, consulted front to back.
    #[must_use]
    pub fn new(layers: Vec<Arc<dyn VirtualFs>>) -> Arc<Self> {
        Arc::new(Self { layers })
    }
}

impl VirtualFs for OverlayFs {
    fn open(&self, path: &Path, opts: &OpenOptions) -> Option<io::Result<Box<dyn FileStream>>> {
        self.layers.iter().find_map(|layer| layer.open(path, opts))
    }

    fn metadata(&self, path: &Path) -> Option<io::Result<VfsMetadata>> {
        self.layers.iter().find_map(|layer| layer.metadata(path))
    }

    fn read_dir(&self, path: &Path) -> Option<io::Result<Vec<VfsDirEntry>>> {
        self.layers.iter().find_map(|layer| layer.read_dir(path))
    }

    fn exists(&self, path: &Path) -> Option<bool> {
        self.layers.iter().find_map(|layer| layer.exists(path))
    }

    fn open_write(
        &self,
        path: &Path,
        opts: &OpenOptions,
    ) -> Option<io::Result<Box<dyn WritableFile>>> {
        self.layers
            .iter()
            .find_map(|layer| layer.open_write(path, opts))
    }

    fn remove(&self, path: &Path) -> Option<io::Result<()>> {
        self.layers.iter().find_map(|layer| layer.remove(path))
    }

    fn rename(&self, from: &Path, to: &Path) -> Option<io::Result<()>> {
        self.layers.iter().find_map(|layer| layer.rename(from, to))
    }

    fn create_dir(&self, path: &Path) -> Option<io::Result<()>> {
        self.layers.iter().find_map(|layer| layer.create_dir(path))
    }

    fn set_len(&self, path: &Path, len: u64) -> Option<io::Result<()>> {
        self.layers
            .iter()
            .find_map(|layer| layer.set_len(path, len))
    }

    fn volume_info(&self) -> Option<VolumeInfo> {
        self.layers.iter().find_map(|layer| layer.volume_info())
    }
}
