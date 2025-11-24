# Lexon Compiler Stable Diffusion Example
1. Enter Python virtual environment

We recommend you to use anaconda3 to create python virtual environment. You should install python packages as lexon-mlir/requirements.

```
$ conda activate <your virtual environment name>
$ cd lexon-mlir
$ pip install -r requirements.txt
```

2. Build and check LLVM/MLIR

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

3. Build and check lexon-mlir

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

Set the `PYTHONPATH` environment variable. Make sure that the `PYTHONPATH` variable includes the directory of LLVM/MLIR python bindings and the directory of Lexon MLIR python packages.

```
$ export PYTHONPATH=/path-to-lexon-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-lexon-mlir/build/python_packages:${PYTHONPATH}

// For example:
// Navigate to your lexon-mlir/build directory
$ cd lexon-mlir/build
$ export LEXON_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${LEXON_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

6. Build and run Stable Diffusion example

```
$ cmake -G Ninja .. -DLEXON_STABLE_DIFFUSION_EXAMPLES=ON
$ ninja lexon-stable-diffusion-run
$ cd bin
$ ./lexon-stable-diffusion-run
```
This build will spend a few minutes. We recommend you to use better cpu such as server-level cpu to run lexon-stable-diffusion-run.

If you wish to utilize `mimalloc` as a memory allocator, you need to set `LEXON_MLIR_USE_MIMALLOC` and `MIMALLOC_BUILD_DIR`.
For more details, please see [here](../../thirdparty/README.md#the-mimalloc-allocator).

