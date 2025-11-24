set(LLVM_MIRParser_DIR ${LLVM_MAIN_SRC_DIR}/lib/CodeGen/MIRParser)

add_llvm_component_library(LLVMLexonMIRParser
  ${LLVM_MIRParser_DIR}/MILexer.cpp
  ${LLVM_MIRParser_DIR}/MIParser.cpp
  ${LLVM_MIRParser_DIR}/MIRParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CodeGen/MIRParser

  DEPENDS
  lexon_intrinsics_gen

  LINK_COMPONENTS
  LexonAsmParser
  BinaryFormat
  LexonCodeGen
  LexonCore
  MC
  Support
  LexonTarget
  )
