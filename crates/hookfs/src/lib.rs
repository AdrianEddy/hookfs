//! `hookfs` routes native file-system calls for selected paths to a Rust virtual
//! filesystem.
//!
//! It preserves native behavior for paths outside the mounted namespace and uses
//! `plthook` to install the Windows, Linux, macOS, and iOS/iPadOS ABI shims. The
//! public VFS traits and providers are platform-independent; installation reports
//! an unavailable backend on targets without one.

mod dispatch;
mod error;
mod install;
mod mount;
mod namespace;
mod providers;
mod registry;
mod router;
mod shims;
pub mod vfs;

pub use error::{Error, Result};
pub use install::{InstallGuard, Options, Scope, install};
pub use mount::{Hookfs, MountGuard, MountedPath, mount_read_seek};
pub use providers::{MemoryFs, OverlayFs};
pub use vfs::{FileStream, OpenOptions, VfsDirEntry, VfsMetadata, VirtualFs};
