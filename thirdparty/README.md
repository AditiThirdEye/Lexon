# Third-Party Dependencies

## The `mimalloc` Allocator

### Build `mimalloc`

```
$ cd lexon-mlir/thirdparty/mimalloc
$ mkdir -p out/release
$ cd out/release
$ cmake -G Ninja ../.. -DCMAKE_INSTALL_PREFIX=$PWD
$ ninja install
```

### Use `mimalloc` in lexon-mlir

Assign the `mimalloc` installation address to the `LEXON_MLIR_USE_MIMALLOC` CMake variable.

For example:

```
$ cd lexon-mlir/build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLEXON_MLIR_USE_MIMALLOC=ON \
    -DMIMALLOC_BUILD_DIR=$PWD/../thirdparty/mimalloc/out/release
```

In the `CMakeLists.txt` file, link shared or static library by using:

```
target_link_libraries(myapp PUBLIC mimalloc) 
target_link_libraries(myapp PUBLIC mimalloc-static)
```
