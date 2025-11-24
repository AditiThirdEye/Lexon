add_mlir_dialect_library(LexonGemminiTransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  LexonGemmini
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
