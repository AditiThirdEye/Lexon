#!/bin/bash
LEXON_OPT := ../../build/bin/lexon-opt

bud-array-attr-lower:
	@${LEXON_OPT} ./bud-array-attr.mlir \
		--lower-bud -o ./log.mlir

bud-constant-lower:
	@${LEXON_OPT} ./bud-constant.mlir \
		--lower-bud -o ./log.mlir

bud-print-lower:
	@${LEXON_OPT} ./bud-print.mlir \
		--lower-bud -o ./log.mlir

bud-str-attr-lower:
	@${LEXON_OPT} ./bud-str-attr.mlir \
		--lower-bud -o ./log.mlir

bud-inline-interface-lower:
	@${LEXON_OPT} ./bud-inline-interface.mlir \
		--inline -o ./log.mlir
