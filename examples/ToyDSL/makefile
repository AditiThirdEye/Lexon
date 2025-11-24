#!/bin/bash
LEXON_OPT := ../../build/bin/lexon-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
MLIR_TOYC := ../../llvm/build/bin/toyc-ch7
LEXON_TOY_DSL := ../../build/bin/lexon-toy-dsl

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

toyc-constant-run:
	@${MLIR_TOYC} ./constant.toy -emit=jit

lexon-toy-constant-ast:
	@${LEXON_TOY_DSL} ./constant.toy  -emit=ast

lexon-toy-constant-mlir:
	@${LEXON_TOY_DSL} ./constant.toy  -emit=mlir

lexon-toy-constant-affine:
	@${LEXON_TOY_DSL} ./constant.toy  -emit=mlir-affine

lexon-toy-constant-llvm:
	@${LEXON_TOY_DSL} ./constant.toy  -emit=mlir-llvm

lexon-toy-constant-translate:
	@${LEXON_TOY_DSL} ./constant.toy  -emit=llvm

lexon-toy-constant-run:
	@${LEXON_TOY_DSL} ./constant.toy  -emit=jit

toyc-function-run:
	@${MLIR_TOYC} ./function.toy -emit=jit

lexon-toy-function-ast:
	@${LEXON_TOY_DSL} ./function.toy  -emit=ast

lexon-toy-function-mlir:
	@${LEXON_TOY_DSL} ./function.toy  -emit=mlir

lexon-toy-function-affine:
	@${LEXON_TOY_DSL} ./function.toy  -emit=mlir-affine

lexon-toy-function-llvm:
	@${LEXON_TOY_DSL} ./function.toy  -emit=mlir-llvm

lexon-toy-function-translate:
	@${LEXON_TOY_DSL} ./function.toy  -emit=llvm

lexon-toy-function-run:
	@${LEXON_TOY_DSL} ./function.toy  -emit=jit

toyc-struct-run:
	@${MLIR_TOYC} ./struct.toy -emit=jit

lexon-toy-struct-ast:
	@${LEXON_TOY_DSL} ./struct.toy  -emit=ast

lexon-toy-struct-mlir:
	@${LEXON_TOY_DSL} ./struct.toy  -emit=mlir

lexon-toy-struct-affine:
	@${LEXON_TOY_DSL} ./struct.toy  -emit=mlir-affine

lexon-toy-struct-llvm:
	@${LEXON_TOY_DSL} ./struct.toy  -emit=mlir-llvm

lexon-toy-struct-translate:
	@${LEXON_TOY_DSL} ./struct.toy  -emit=llvm

lexon-toy-struct-run:
	@${LEXON_TOY_DSL} ./struct.toy  -emit=jit
