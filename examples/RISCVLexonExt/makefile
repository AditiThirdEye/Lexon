#!/bin/bash
LEXON_LLC := ../../build/bin/lexon-llc

rv-lexon-mvin-mvout-asm:
	@${LEXON_LLC} ./rv-lexon-mvin-mvout.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-mvin-mvout-obj:
	@${LEXON_LLC} ./rv-lexon-mvin-mvout.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-mvin-mvout-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvin-mvout-pk

rv-lexon-mvin-mvout-run:
	spike --extension=gemmini pk mvin-mvout-pk

rv-lexon-matrix-add-asm:
	@${LEXON_LLC} ./rv-lexon-matrix-add.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-matrix-add-obj:
	@${LEXON_LLC} ./rv-lexon-matrix-add.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-matrix-add-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matrix-add-pk

rv-lexon-matrix-add-run:
	spike --extension=gemmini pk matrix-add-pk

rv-lexon-transpose-asm:
	@${LEXON_LLC} ./rv-lexon-transpose.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-transpose-obj:
	@${LEXON_LLC} ./rv-lexon-transpose.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-transpose-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o transpose-pk

rv-lexon-transpose-run:
	spike --extension=gemmini pk transpose-pk

rv-lexon-mvin-padded-asm:
	@${LEXON_LLC} ./rv-lexon-mvin-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-mvin-padded-obj:
	@${LEXON_LLC} ./rv-lexon-mvin-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-mvin-padded-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvin-padded-pk

rv-lexon-mvin-padded-run:
	spike --extension=gemmini pk mvin-padded-pk

rv-lexon-mvout-padded-asm:
	@${LEXON_LLC} ./rv-lexon-mvout-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-mvout-padded-obj:
	@${LEXON_LLC} ./rv-lexon-mvout-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-mvout-padded-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvout-padded-pk

rv-lexon-mvout-padded-run:
	spike --extension=gemmini pk mvout-padded-pk

rv-lexon-matmuls-padded-asm:
	@${LEXON_LLC} ./rv-lexon-matmuls-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-matmuls-padded-obj:
	@${LEXON_LLC} ./rv-lexon-matmuls-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-matmuls-padded-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matmuls-padded-pk

rv-lexon-matmuls-padded-run:
	spike --extension=gemmini pk matmuls-padded-pk

rv-lexon-tile-matmul-ws-asm:
	@${LEXON_LLC} ./rv-lexon-tile-matmul-ws.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-tile-matmul-ws-obj:
	@${LEXON_LLC} ./rv-lexon-tile-matmul-ws.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-tile-matmul-ws-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o tile-matmul-ws-pk -lm

rv-lexon-tile-matmul-ws-run:
	spike --extension=gemmini pk tile-matmul-ws-pk

rv-lexon-conv-asm:
	@${LEXON_LLC} ./rv-lexon-conv.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext \
		-o ./log.s

rv-lexon-conv-obj:
	@${LEXON_LLC} ./rv-lexon-conv.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+lexonext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-lexon-conv-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o conv-pk -lm

rv-lexon-conv-run:
	spike --extension=gemmini pk conv-pk
