# LEXON MLIR

An MLIR-based compiler framework designed for a co-design ecosystem from DSL (domain-specific languages) to DSA (domain-specific architectures). ([Project page](https://lexon-compiler.github.io/))

## Getting Started

### LLVM/MLIR Dependencies

Please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available
on your machine.

### Clone and Initialize

```
$ git clone git@github.com:lexon-compiler/lexon-mlir.git
$ cd lexon-mlir
$ git submodule update --init llvm
```

### Prepare Python Environment

```
$ conda activate <your virtual environment name>
$ cd lexon-mlir
$ pip install -r requirements.txt
```

### Build and Test LLVM/MLIR/CLANG

```
$ cd lexon-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir omp
```

If your target machine includes an NVIDIA GPU, you can add the following configuration:

```
-DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
-DMLIR_ENABLE_CUDA_RUNNER=ON \
```

### Build lexon-mlir

```
$ cd lexon-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLEXON_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-lexon
```

Set the `PYTHONPATH` environment variable to include both the LLVM/MLIR Python bindings and `lexon-mlir` Python packages:

```
$ export LEXON_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${LEXON_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

If you want to test your model end-to-end conversion and inference, you can add the following configuration

```
$ cmake -G Ninja .. -DLEXON_ENABLE_E2E_TESTS=ON
$ ninja check-e2e
```

## Examples

We provide examples to demonstrate how to use the passes and interfaces in `lexon-mlir`, including IR-level transformations, domain-specific applications, and testing demonstrations.

For more details, please see the [examples documentation](./examples/README.md).

## Contributions

We welcome contributions to our open-source project!

Before contributing, please read the [Contributor Guide](https://lexoncompiler.com/Pages/ContributorGuide.html) and [Code Style](https://lexoncompiler.com/Pages/Documentation/CodeStyle.html).

To maintain code quality, this project provides pre-commit checks:

```
$ pre-commit install
```

For direct access to the paper, please visit [Compiler Technologies in Deep Learning Co-Design: A Survey](https://spj.science.org/doi/10.34133/icomputing.0040).
