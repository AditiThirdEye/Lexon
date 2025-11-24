set(LLVM_Reader_DIR ${LLVM_MAIN_SRC_DIR}/lib/Bitcode/Reader)

add_llvm_component_library(LLVMLexonBitReader
  ${LLVM_Reader_DIR}/BitcodeAnalyzer.cpp
  ${LLVM_Reader_DIR}/BitReader.cpp
  ${LLVM_Reader_DIR}/BitcodeReader.cpp
  ${LLVM_Reader_DIR}/MetadataLoader.cpp
  ${LLVM_Reader_DIR}/ValueList.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Bitcode

  DEPENDS
  lexon_intrinsics_gen

  LINK_COMPONENTS
  BitstreamReader
  LexonCore
  Support
  TargetParser
  )

