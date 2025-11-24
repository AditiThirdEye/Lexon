# 定义路径
LEXON_OPT = /home/wuxintong/lexon-mlir/build/bin/lexon-opt
FILECHECK = /home/wuxintong/lexon-mlir/llvm/build/bin/FileCheck
TEST_FILE = /home/wuxintong/lexon-mlir/tests/Dialect/LexonGPU/hoist-static-alloc.mlir

# 默认目标
all: test

# 测试命令
test:
	$(LEXON_OPT) --split-input-file --transform-interpreter $(TEST_FILE) | $(FILECHECK) $(TEST_FILE)

# 清理命令（如果需要）
clean:
	@echo "Nothing to clean for this test."
