set(LLVM_Writer_DIR ${LLVM_MAIN_SRC_DIR}/lib/Bitcode/Writer)


add_llvm_component_library(LLVMLexonBitWriter
  ${LLVM_Writer_DIR}/BitWriter.cpp
  ${LLVM_Writer_DIR}/BitcodeWriter.cpp
  ${LLVM_Writer_DIR}/BitcodeWriterPass.cpp
  ${LLVM_Writer_DIR}/ValueEnumerator.cpp

  DEPENDS
  lexon_intrinsics_gen

  LINK_COMPONENTS
  Analysis
  LexonCore
  MC
  LexonObject
  Object
  Support
  TargetParser
  )
