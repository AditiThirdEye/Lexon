# Alternative Build Methods

This document describes additional build configurations and alternative methods for building `lexon-mlir` beyond the standard approach outlined in the main README.

## Table of Contents

- [Build with Image Processing Libraries](#build-with-image-processing-libraries)
- [One-Step Build Strategy](#one-step-build-strategy)
- [Build with Nix](#build-with-nix)
- [Tools](#tools)

## Build with Image Processing Libraries

To configure the build environment for using image processing libraries, add the following options to your cmake configuration:

```bash
$ cd lexon-mlir/build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLEXON_MLIR_ENABLE_DIP_LIB=ON \
    -DLEXON_ENABLE_PNG=ON
$ ninja
$ ninja check-lexon
```

**Configuration Options:**

- `LEXON_MLIR_ENABLE_DIP_LIB=ON`: Enables the Digital Image Processing (DIP) library
- `LEXON_ENABLE_PNG=ON`: Enables PNG format support

## One-Step Build Strategy

If you want to use `lexon-mlir` tools and integrate them more easily into your projects, you can use the one-step build strategy. This method builds LLVM, MLIR, and `lexon-mlir` together as an LLVM external project.

```bash
$ cd lexon-mlir
$ cmake -G Ninja -Bbuild \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_EXTERNAL_PROJECTS="lexon-mlir" \
    -DLLVM_EXTERNAL_LEXON_MLIR_SOURCE_DIR="$PWD" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    llvm/llvm
$ cd build
$ ninja check-mlir check-clang
$ ninja
$ ninja check-lexon
```

## Build with Nix

This repository provides Nix flake support for reproducible builds. Follow the [Nix installation instructions](https://nixos.org/manual/nix/stable/installation/installation.html) and enable [flake features](https://nixos.wiki/wiki/Flakes#Other_Distros.2C_without_Home-Manager) to set up Nix on your system.

### Development Environment

If you want to contribute to this project, enter the development shell:

```bash
$ nix develop .
```

This command sets up a bash shell with `clang`, `ccls`, `cmake`, `ninja`, and other necessary dependencies to build `lexon-mlir` from source.

### Binary Tools

If you only want to use the `lexon-mlir` binary tools:

```bash
$ nix build .#lexon-mlir
$ ./result/bin/lexon-opt --version
```

This approach provides a fully isolated and reproducible build environment without affecting your system configuration.

## Tools

### lexon-opt

`lexon-opt` is the optimization driver for `lexon-mlir`, similar to `mlir-opt` in LLVM. It provides access to all dialects and optimization passes defined in the `lexon-mlir` project.

**Usage:**

```bash
$ lexon-opt [options] <input-file>
```

**Common Options:**
- `--help`: Display available passes and options
- `--pass-pipeline`: Specify a custom pass pipeline
- `--mlir-print-ir-after-all`: Print IR after each pass

**Example:**

```bash
$ lexon-opt --lower-affine --convert-linalg-to-loops input.mlir
```

### lexon-lsp-server

`lexon-lsp-server` is a drop-in replacement for `mlir-lsp-server`, providing Language Server Protocol (LSP) support for all dialects defined in `lexon-mlir`.

**Features:**
- Code completion for custom dialects (`rvv`, `gemmini`, `dip`, etc.)
- Real-time error diagnostics
- Hover information and symbol navigation
- Syntax highlighting support

**Setup for VSCode:**

Modify the MLIR LSP server path in your VSCode settings:

```json
{
    "mlir.server_path": "/path/to/lexon-mlir/build/bin/lexon-lsp-server"
}
```
