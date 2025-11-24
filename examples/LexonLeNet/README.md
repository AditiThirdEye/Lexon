# Lexon Compiler LeNet Example

## Train the LeNet Model

Activate your python environment.

```bash
$ cd lexon-mlir
$ cd examples/LexonLeNet
$ python pytorch-lenet-train.py
```

## LeNet Model Inference

### Activate your python environment.

```bash
$ conda activate <your env>
```

### Build LLVM

```bash
$ cd lexon-mlir
$ mkdir llvm/build
$ cd llvm/build

// CPU
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)

// GPU
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)

$ ninja check-clang check-mlir omp
```

### Build lexon-mlir

```bash
$ cd lexon-mlir
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLEXON_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DLEXON_MLIR_ENABLE_DIP_LIB=ON \
    -DLEXON_ENABLE_PNG=ON
$ ninja
$ ninja check-lexon
```

### Set the `PYTHONPATH` environment variable.

Make sure you are in the build directory.

```bash
$ export LEXON_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${LEXON_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

### Build and run the LeNet example

```bash
$ cmake -G Ninja .. -DLEXON_LENET_EXAMPLES=ON

// CPU
$ ninja lexon-lenet-run
$ cd bin
$ ./lexon-lenet-run

// GPU
$ ninja lexon-lenet-run-gpu
$ cd bin
$ ./lexon-lenet-run-gpu
```

## Debug the Lowering Pass Pipeline with Fake Parameters.

```bash
$ cd lexon-mlir
$ cd examples/LexonLeNet
$ make lexon-lenet-lower
$ make lexon-lenet-translate
$ make lexon-lenet-run
```
