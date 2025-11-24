add_mlir_dialect_library(LexonGemmini
  GemminiDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
