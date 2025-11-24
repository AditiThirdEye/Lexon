add_mlir_translation_library(LexonGemminiToLLVMIRTranslation
  GemminiToLLVMIRTranslation.cpp

  DEPENDS
  LexonGemminiConversionsIncGen
  lexon_intrinsics_gen
  
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  LexonGemmini
  )
