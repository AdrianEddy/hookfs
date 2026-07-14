# hookfs

`hookfs` lets a native library open Rust-owned streams and virtual files through
ordinary absolute paths without materializing those files on disk. It patches
the selected module's imported file-system functions and routes only paths in a
private synthetic namespace to a Rust `VirtualFs`; every other path is passed to
the original operating-system implementation.

This is useful when a closed-source SDK insists on receiving a file name but the
application owns the data in memory, an archive, a remote object store, or
another custom storage layer.

## Features

- Mount any owned `Read + Seek + Send + 'static` stream.
- Give every native open an independent cursor over the shared source.
- Serve metadata, seeking, directory enumeration, and normal read APIs.
- Optionally expose in-memory create, write, truncate, and delete operations.
- Scope hooks to one module, a module allow/deny list, or all loaded modules.
- Rescan late-loaded modules when a native SDK loads plugins on demand.
- Implement a custom `VirtualFs`, or compose providers with `OverlayFs`.
- Restore import slots automatically when the installation guard is dropped.

## Installation

```toml
[dependencies]
hookfs = "0.1.0"
```

The minimum supported Rust version is 1.96.

## Mounting a stream

The normal workflow is to create one context, mount one or more streams, install
the hooks, and pass the resulting synthetic paths to the native library.
`Options::for_module` needs an address only to locate the module to patch. It may
be the address of any export from that exact DLL, `.so`, or `.dylib`, typically
an SDK entry point the application already resolved. For example, a BRAW caller
can use `CreateBlackmagicRawFactoryInstance`:

```rust
use hookfs::{Hookfs, Options};
use std::ffi::c_void;
use std::io::Cursor;
use std::path::Path;

fn main() -> hookfs::Result<()> {
    let fs = Hookfs::new();
    let clip = fs.mount("clip.bin", Cursor::new(vec![1, 2, 3, 4]))?;

    // The address must belong to the native module that will issue file calls.
    let hooks = fs.install(Options::for_module(sdk_module_anchor()))?;

    sdk_open(&fs.path_for("clip.bin"));

    // Drop hooks before mounts, and keep both alive until native work is finished.
    drop(hooks);
    drop(clip);
    Ok(())
}

fn sdk_module_anchor() -> *const c_void {
    // Return any export address from the SDK module to patch. For example:
    // GetProcAddress(sdk, "CreateBlackmagicRawFactoryInstance") on Windows, or
    // dlsym(sdk, "CreateBlackmagicRawFactoryInstance") on Unix.
    todo!("return an export address from the target SDK module")
}

fn sdk_open(_path: &Path) {
    // Pass the synthetic path to the SDK.
}
```

For a single stream, `mount_read_seek` returns a `MountedPath` convenience value
that owns its context and mount guard.

Mount names must be relative and lexically safe. Absolute names, parent
traversal, NUL bytes, and Windows drive/device/stream syntax are rejected.

## Writable virtual files

Writes are fail-closed by default. Enable them explicitly when the native
library must create a sidecar or output file:

```rust
use hookfs::{Hookfs, Options};
use std::ffi::c_void;

fn main() -> hookfs::Result<()> {
    let fs = Hookfs::with_capacity(64 * 1024 * 1024);
    let output = fs.mount_writable("result.bin")?;
    let hooks = fs.install(
        Options::for_module(sdk_module_anchor()).allow_writes(true),
    )?;

    // Call the SDK with fs.path_for("result.bin"), then copy the result back.
    let bytes = fs.read_virtual("result.bin");

    drop(hooks);
    drop(output);
    println!("the SDK wrote {} bytes", bytes.map_or(0, |data| data.len()));
    Ok(())
}

fn sdk_module_anchor() -> *const c_void {
    // Any exported function in the exact SDK module that will open the file.
    // The address identifies the module; hookfs does not call this function.
    todo!("return an export address from the target SDK module")
}
```

`with_capacity` controls the synthetic volume capacity. Provider errors are
translated to the platform ABI, and panics are contained at the FFI boundary.

## Hook scope

`Options::for_module(address)` is the safest default: only the module containing
the supplied live address is patched. Other choices are available through
`Scope`:

- `Scope::AllModules` patches all supported modules currently loaded.
- `Scope::Only(names)` patches matching module file names.
- `Scope::Exclude(names)` patches everything except matching file names.
- `Options::auto_rescan(true)` also hooks supported modules loaded later.

Only one `hookfs` installation can be active in a process at a time. Drop the
existing `InstallGuard` before installing another one. `InstallGuard::rescan`
can be used for an explicit late-module scan.

## Custom providers

Implement `VirtualFs` when files should come from application-specific storage.
Read operations return `Option<io::Result<_>>`: `None` means the path is not
owned by that provider, allowing normal passthrough or another overlay layer.
Mutation methods have the same not-owned convention and are called only when
writes are enabled.

Use `hookfs::install(Arc<dyn VirtualFs>, Options)` for a provider you manage
directly. `MemoryFs` provides the built-in in-memory tree, while `OverlayFs`
queries multiple providers in first-match order.

## Platform support and limitations

Live hooks support Windows/PE32+, Linux/glibc ELF64, macOS Mach-O 64, and
iOS/iPadOS Mach-O 64 on `x86_64` or `aarch64` where the OS supports that
architecture. Other targets retain the platform-independent types but return
`Error::UnsupportedPlatform` from installation.

The mechanism redirects imported calls in patched modules. It cannot intercept
inlined functions, direct syscalls, static linkage, or calls through modules
outside the chosen scope. It is not a sandbox or security boundary. Do not
unload a target native module or release mounts while it can still perform file
operations.

## License

Original work is licensed under either MIT or Apache-2.0, at your option.
`hookfs` depends on `plthook-native`, a Rust port of
[`kubo/plthook`](https://github.com/kubo/plthook); the dependency carries its
upstream BSD-2-Clause notice.
