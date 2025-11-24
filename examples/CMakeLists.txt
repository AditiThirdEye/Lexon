if(LEXON_EXAMPLES)
  add_subdirectory(ConvOpt)
  add_subdirectory(DIPDialect)
  add_subdirectory(DAPDialect)
endif()

if(LEXON_LLAMA_EXAMPLES)
  add_subdirectory(LexonLlama)
endif()

if(LEXON_DEEPSEEKR1_EXAMPLES)
  add_subdirectory(LexonDeepSeekR1)
endif()

if(LEXON_TRANSFORMER_EXAMPLES)
  add_subdirectory(LexonTransformer)
endif()

if (LEXON_BERT_EXAMPLES)
  add_subdirectory(LexonBert)
endif()

if (LEXON_LENET_EXAMPLES)
  add_subdirectory(LexonLeNet)
endif()

if(LEXON_WHISPER_EXAMPLES)
  add_subdirectory(LexonWhisper)
endif()

if (LEXON_MOBILENETV3_EXAMPLES)
  add_subdirectory(LexonMobileNetV3)
endif()

if (LEXON_RESNET_EXAMPLES)
  add_subdirectory(LexonResNet18)
endif()

if (LEXON_STABLE_DIFFUSION_EXAMPLES)
  add_subdirectory(LexonStableDiffusion)
endif()

if (LEXON_ONEDNN_EXAMPLES)
  add_subdirectory(LexonOneDNN)
endif()

if(LEXON_DSL_EXAMPLES)
  add_subdirectory(ToyDSL)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LEXON_EXAMPLES_DEPENDS
  FileCheck count not
  lexon-opt
  lexon-translate
  mlir-runner
  )

if(LEXON_MLIR_ENABLE_PYTHON_PACKAGES)
  list(APPEND LEXON_TEST_DEPENDS LexonMLIRPythonModules)
endif()

add_lit_testsuite(check-examples "Checking the lexon-mlir examples..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LEXON_EXAMPLES_DEPENDS}
  )
set_target_properties(check-examples PROPERTIES FOLDER "Examples")

add_lit_testsuites(LEXON-EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${LEXON_EXAMPLES_DEPENDS})
