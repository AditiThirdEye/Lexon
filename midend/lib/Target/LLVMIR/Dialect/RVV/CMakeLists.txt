add_mlir_translation_library(LexonRVVToLLVMIRTranslation
  RVVToLLVMIRTranslation.cpp

  DEPENDS
  LexonRVVConversionsIncGen
  lexon_intrinsics_gen
  
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  LexonRVV
  )
