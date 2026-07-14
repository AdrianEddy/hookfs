//! Batteries-included [`VirtualFs`](crate::vfs::VirtualFs) providers.

mod memory;
mod overlay;

pub use memory::MemoryFs;
pub use overlay::OverlayFs;
