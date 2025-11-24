set(LLVM_LINK_COMPONENTS
  LexonRISCVCodeGen
  LexonRISCVAsmParser
  LexonRISCVDesc
  LexonRISCVInfo

  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LexonAnalysis
  LexonAsmParser
  LexonAsmPrinter
  LexonCodeGen
  LexonCore
  LexonIRReader
  MC
  LexonMIRParser
  LexonRemarks
  LexonScalarOpts
  LexonSelectionDAG
  Support
  LexonTarget
  LexonIPO
  LexonVectorize
  TargetParser
  LexonTransformUtils
  Passes
)

add_llvm_tool(lexon-llc
  ${LLVM_MAIN_SRC_DIR}/tools/llc/llc.cpp
  ${LLVM_MAIN_SRC_DIR}/tools/llc/NewPMDriver.cpp

  DEPENDS
  lexon_intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(lexon-llc)
