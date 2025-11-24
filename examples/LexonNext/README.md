# Next Exploration Steps in Lexon MLIR

## Build llvm/mlir

```
$ cd lexon-mlir/llvm
$ mkdir build && cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja check-clang check-mlir omp
```
Some test errors may occur when running check-mlir on RISC-V platforms, please ignore them.


## Build lexon-mlir

Build MLIR/LLVM toolchains and libraries

```
$ cd lexon-mlir
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
$ ninja
```

## Run the examples

```
$ cd examples/MLIRNext
$ make <target name> (e.g. `make next-mhsa-core-aot-omp`)
```
