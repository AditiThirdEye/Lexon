add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertLexonToLLVMIR.cpp
  )

add_mlir_translation_library(LexonToLLVMIRTranslationRegistration
  ConvertLexonToLLVMIR.cpp

  LINK_LIBS PUBLIC
  LexonRVVToLLVMIRTranslation
  LexonGemminiToLLVMIRTranslation
  )
