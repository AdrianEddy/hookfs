//! Live PE module acquisition and import enumeration.
//!
//! A [`Module`] is a validated identity for one image currently mapped into the
//! process: its base address, its loader-reported mapped size (the bound every
//! parser read is checked against), and its path. Acquiring one validates the
//! PE headers up front, so an invalid handle/address is rejected here rather than
//! deep inside enumeration.

use crate::error::{Error, Result};
use crate::pe::{self, MappedImage};
use crate::slot::ImportSlot;
use crate::sys;
use core::ffi::c_void;

/// A validated, currently-loaded module.
#[derive(Debug, Clone)]
pub struct Module {
    base: usize,
    size: usize,
    path: String,
}

impl Module {
    /// Acquire a module from its `HMODULE`.
    ///
    /// # Safety
    /// `handle` must be a module handle currently valid in this process (as
    /// returned by `LoadLibrary*`/`GetModuleHandle*`). It is validated against
    /// the loader before use, but a stale/foreign handle is still unsound.
    pub unsafe fn from_handle(handle: *mut c_void) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::ModuleNotFound {
                reason: "null handle",
                os_error: 0,
            });
        }
        Self::from_base(handle as usize)
    }

    /// Acquire the module containing `address`, such as an exported
    /// `CreateBlackmagicRawFactoryInstance` function. Uses
    /// `GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS)`.
    ///
    /// # Safety
    /// `address` must point into code or data of a currently-loaded module.
    pub unsafe fn from_address(address: *const c_void) -> Result<Self> {
        let base = sys::module_from_address(address)?;
        Self::from_base(base)
    }

    /// Acquire an already-loaded module by name (`GetModuleHandleW`). Safe: the
    /// loader validates the name and never changes the module's refcount.
    pub fn from_name(name: &str) -> Result<Self> {
        let base = sys::module_from_name(name)?;
        Self::from_base(base)
    }

    /// Enumerate every module loaded in the process (`Scope::AllModules` /
    /// `EnumProcessModules`). Images that fail validation are skipped.
    pub fn enumerate() -> Result<Vec<Self>> {
        Ok(sys::enumerate_modules()?
            .into_iter()
            .filter_map(|base| Self::from_base(base).ok())
            .collect())
    }

    /// Validate headers and capture identity for a module base address.
    fn from_base(base: usize) -> Result<Self> {
        let size = sys::image_size(base)?;
        // SAFETY: `base`/`size` come from the loader for a mapped module.
        let image = unsafe { MappedImage::from_loaded(base, size) };
        pe::validate(&image)?;
        Ok(Self {
            base,
            size,
            path: sys::module_path(base),
        })
    }

    /// File name of the module (final path component).
    #[must_use]
    pub fn name(&self) -> &str {
        let bytes = self.path.as_bytes();
        match bytes.iter().rposition(|&b| b == b'\\' || b == b'/') {
            Some(pos) => self.path.get(pos + 1..).unwrap_or(&self.path),
            None => &self.path,
        }
    }

    /// Full filesystem path of the module.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Base address (`HMODULE`) of the mapped image.
    #[must_use]
    pub fn base(&self) -> *const c_void {
        self.base as *const c_void
    }

    /// Loader-reported mapped size (`SizeOfImage`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn base_addr(&self) -> usize {
        self.base
    }

    /// Enumerate the module's import slots (standard + delay-load).
    ///
    /// Every returned slot's address is bounds-checked to lie inside the mapped
    /// image, so it is safe to pass to [`crate::install`].
    pub fn imports(&self) -> Result<Vec<ImportSlot>> {
        // SAFETY: `base`/`size` describe the currently-mapped image.
        let image = unsafe { MappedImage::from_loaded(self.base, self.size) };
        Ok(pe::parse_imports(&image)?
            .into_iter()
            .map(|raw| ImportSlot::from_raw(self.base, raw))
            .collect())
    }
}
