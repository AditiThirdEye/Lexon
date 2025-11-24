#!/bin/bash
LEXON_OPT := ../../build/bin/lexon-opt
LEXON_TRANSLATE := ../../build/bin/lexon-translate
LEXON_LLC := ../../build/bin/lexon-llc
INTERFACES:= ../../frontend/Interfaces

mvin-mvout-lower:
	@${LEXON_OPT} ./mvin-mvout.mlir \
		-lower-gemmini \
		-o log.mlir

mvin-mvout-translate:
	@${LEXON_OPT} ./mvin-mvout.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

mvin-mvout-asm:
	@${LEXON_OPT} ./mvin-mvout.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

mvin-mvout-run:
	@${LEXON_OPT} ./mvin-mvout.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-lower:
	@${LEXON_OPT} ./matrix-add.mlir \
		-lower-gemmini \
		-o log.mlir

matrix-add-translate:
	@${LEXON_OPT} ./matrix-add.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

matrix-add-asm:
	@${LEXON_OPT} ./matrix-add.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

matrix-add-run:
	@${LEXON_OPT} ./matrix-add.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-scale-lower:
	@${LEXON_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini \
		-o log.mlir

matrix-add-scale-translate:
	@${LEXON_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

matrix-add-scale-asm:
	@${LEXON_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

matrix-add-scale-run:
	@${LEXON_OPT} ./matrix-add-scale.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

transpose-lower:
	@${LEXON_OPT} ./transpose.mlir \
		-lower-gemmini \
		-o log.mlir

transpose-translate:
	@${LEXON_OPT} ./transpose.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

transpose-asm:
	@${LEXON_OPT} ./transpose.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

transpose-run:
	@${LEXON_OPT} ./transpose.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-os-lower:
	@${LEXON_OPT} ./matmul-os.mlir \
		-lower-gemmini \
		-o log.mlir

matmul-os-translate:
	@${LEXON_OPT} ./matmul-os.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

matmul-os-asm:
	@${LEXON_OPT} ./matmul-os.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

matmul-os-run:
	@${LEXON_OPT} ./matmul-os.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

compute-accumulated-lower:
	@${LEXON_OPT} ./compute-accumulated.mlir \
		-lower-gemmini \
		-o log.mlir

compute-accumulated-translate:
	@${LEXON_OPT} ./compute-accumulated.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

compute-accumulated-asm:
	@${LEXON_OPT} ./compute-accumulated.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

compute-accumulated-run:
	@${LEXON_OPT} ./compute-accumulated.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-ws-lower:
	@${LEXON_OPT} ./matmul-ws.mlir \
		-lower-gemmini \
		-o log.mlir

matmul-ws-translate:
	@${LEXON_OPT} ./matmul-ws.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

matmul-ws-asm:
	@${LEXON_OPT} ./matmul-ws.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

matmul-ws-run:
	@${LEXON_OPT} ./matmul-ws.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-lower:
	@${LEXON_OPT} ./tile-matmul.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-translate:
	@${LEXON_OPT} ./tile-matmul.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-matmul-asm:
	@${LEXON_OPT} ./tile-matmul.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-matmul-run:
	@${LEXON_OPT} ./tile-matmul.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-os-lower:
	@${LEXON_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-os-translate:
	@${LEXON_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-matmul-os-asm:
	@${LEXON_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-matmul-os-run:
	@${LEXON_OPT} ./tile-matmul-os.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-igelu-lower:
	@${LEXON_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-igelu-translate:
	@${LEXON_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-matmul-ws-igelu-asm:
	@${LEXON_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-igelu-run:
	@${LEXON_OPT} ./tile-matmul-ws-igelu.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-relu-lower:
	@${LEXON_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-relu-translate:
	@${LEXON_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-matmul-ws-relu-asm:
	@${LEXON_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-relu-run:
	@${LEXON_OPT} ./tile-matmul-ws-relu.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-softmax-lower:
	@${LEXON_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-softmax-translate:
	@${LEXON_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-matmul-ws-softmax-asm:
	@${LEXON_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-softmax-run:
	@${LEXON_OPT} ./tile-matmul-ws-softmax.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-layernorm-lower:
	@${LEXON_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-layernorm-translate:
	@${LEXON_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-matmul-ws-layernorm-asm:
	@${LEXON_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-layernorm-run:
	@${LEXON_OPT} ./tile-matmul-ws-layernorm.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-lower:
	@${LEXON_OPT} ./tile-conv.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-translate:
	@${LEXON_OPT} ./tile-conv.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-conv-asm:
	@${LEXON_OPT} ./tile-conv.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-conv-run:
	@${LEXON_OPT} ./tile-conv.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-igelu-lower:
	@${LEXON_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-igelu-translate:
	@${LEXON_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-conv-igelu-asm:
	@${LEXON_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-conv-igelu-run:
	@${LEXON_OPT} ./tile-conv-igelu.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-softmax-lower:
	@${LEXON_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-softmax-translate:
	@${LEXON_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-conv-softmax-asm:
	@${LEXON_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-conv-softmax-run:
	@${LEXON_OPT} ./tile-conv-softmax.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-relu-lower:
	@${LEXON_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-relu-translate:
	@${LEXON_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-conv-relu-asm:
	@${LEXON_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-conv-relu-run:
	@${LEXON_OPT} ./tile-conv-relu.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-layernorm-lower:
	@${LEXON_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-layernorm-translate:
	@${LEXON_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-conv-layernorm-asm:
	@${LEXON_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-conv-layernorm-run:
	@${LEXON_OPT} ./tile-conv-layernorm.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-rect-conv-lower:
	@${LEXON_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini \
		-o log.mlir

tile-rect-conv-translate:
	@${LEXON_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

tile-rect-conv-asm:
	@${LEXON_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

tile-rect-conv-run:
	@${LEXON_OPT} ./tile-rect-conv.mlir -lower-gemmini | \
	${LEXON_TRANSLATE} --lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-matmul-lower:
	@${LEXON_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-matmul-translate:
	@${LEXON_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-linalg-matmul-asm:
	@${LEXON_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-matmul-run:
	@${LEXON_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nchw-fchw-i8-lower:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-conv2d-nchw-fchw-i8-translate:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nchw-fchw-i8-asm:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nchw-fchw-i8-run:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nchw-fchw-f32-lower:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" \
		-o log.mlir

gemmini-linalg-conv2d-nchw-fchw-f32-translate:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nchw-fchw-f32-asm:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nchw-fchw-f32-run:
	@${LEXON_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nhwc-hwcf-i8-lower:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-conv2d-nhwc-hwcf-i8-translate:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nhwc-hwcf-i8-asm:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nhwc-hwcf-i8-run:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nhwc-hwcf-f32-lower:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" \
		-o log.mlir

gemmini-linalg-conv2d-nhwc-hwcf-f32-translate:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nhwc-hwcf-f32-asm:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nhwc-hwcf-f32-run:
	@${LEXON_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-batch-matmul-lower:
	@${LEXON_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-batch-matmul-translate:
	@${LEXON_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-linalg-batch-matmul-asm:
	@${LEXON_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-batch-matmul-run:
	@${LEXON_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

linalg-matmul-32x32-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini \
		-o log.mlir

linalg-matmul-32x32-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-32x32-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-32x32-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-32x32-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=1 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-32x32-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-32x32-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-32x32-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-32x32-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-32x32-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-32x32-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-matmul-32x32-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-32x32-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-64x64-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-64x64-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-64x64-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-64x64-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-64x64-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=2 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-64x64-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-64x64-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-64x64-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-64x64-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-64x64-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-64x64-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-matmul-64x64-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-64x64-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-128x128-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-128x128-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-128x128-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-128x128-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-128x128-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=3 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-128x128-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-128x128-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-128x128-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-128x128-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-128x128-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-128x128-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-matmul-128x128-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-128x128-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-256x256-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-256x256-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-256x256-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-256x256-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-256x256-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=4 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-256x256-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-256x256-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-256x256-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-256x256-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-256x256-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-256x256-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-matmul-256x256-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-256x256-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-512x512-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-512x512-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-512x512-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-512x512-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-512x512-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=5 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-512x512-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-512x512-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-512x512-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-512x512-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-512x512-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-512x512-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-matmul-512x512-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-512x512-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-1024x1024-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-1024x1024-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-1024x1024-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-1024x1024-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-1024x1024-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=6 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-1024x1024-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-1024x1024-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-matmul-1024x1024-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-matmul-1024x1024-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-1024x1024-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-1024x1024-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-matmul-1024x1024-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-1024x1024-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-3x3-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-3x3-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-3x3-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-3x3-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-3x3-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=1 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-3x3-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-3x3-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-3x3-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-3x3-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-3x3-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-3x3-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-conv-3x3-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-conv-3x3-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-5x5-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-5x5-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-5x5-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-5x5-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-5x5-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=2 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-5x5-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-5x5-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-5x5-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-5x5-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-5x5-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-5x5-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-conv-5x5-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-conv-5x5-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-7x7-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-7x7-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-7x7-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-7x7-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-7x7-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=3 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-7x7-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-7x7-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-7x7-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-7x7-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-7x7-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-7x7-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-conv-7x7-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-conv-7x7-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-9x9-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-9x9-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-9x9-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-9x9-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-9x9-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=4 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-9x9-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-9x9-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-9x9-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-9x9-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-9x9-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-9x9-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-conv-9x9-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-conv-9x9-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-11x11-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-11x11-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-11x11-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-11x11-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-11x11-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=5 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-11x11-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-11x11-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-11x11-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-11x11-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-11x11-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-11x11-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-conv-11x11-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-conv-11x11-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out


linalg-conv-13x13-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-13x13-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-13x13-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-13x13-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-13x13-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=6 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-13x13-cpu-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-13x13-cpu-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

linalg-conv-13x13-cpu-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

linalg-conv-13x13-cpu-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-13x13-gemmini-lower:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-13x13-gemmini-translate:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir \
		-o log.ll

gemmini-conv-13x13-gemmini-asm:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.s

gemmini-conv-13x13-gemmini-run:
	@${LEXON_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${LEXON_TRANSLATE} -lexon-to-llvmir | \
	${LEXON_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+lexonext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

exo-matmul-4-run:
	@riscv64-unknown-linux-gnu-gcc ./exo-matmul-4.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests \
	-O2 -static -o a.out
	@spike --extension=gemmini pk a.out
