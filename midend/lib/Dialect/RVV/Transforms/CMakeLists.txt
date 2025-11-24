add_mlir_dialect_library(LexonRVVTransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  LexonRVVConversionsIncGen

  LINK_LIBS PUBLIC
  LexonRVV
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
