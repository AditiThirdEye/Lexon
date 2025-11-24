# Lexon Graph Representation Examples

## Run the Examples

0. Enter your Python Env
```
(base)$ conda activate lexon
(lexon)$ ...
```
1. Build Python Packages
2. Configure Python Path
```
(lexon)$ cd lexon-mlir/build
(lexon)$ export LEXON_MLIR_BUILD_DIR=$PWD
(lexon)$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
(lexon)$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${LEXON_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}

```
3. Run the Examples
```
(lexon)$ cd examples/LexonGraph
(lexon)$ python import-dynamo-break.py
```