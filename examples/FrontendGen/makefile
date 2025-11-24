#!/bin/bash
LEXON_FRONTEND_GEN := ../../build/bin/lexon-frontendgen

frontendgen-emit-ast:
	@${LEXON_FRONTEND_GEN} -f ./example.fegen -emit=ast

frontendgen-emit-antlr:
	@${LEXON_FRONTEND_GEN} -f ./example.fegen -emit=antlr -g Toy

frontendgen-emit-visitor:
	@${LEXON_FRONTEND_GEN} -f ./example.fegen -emit=visitor -g Toy

