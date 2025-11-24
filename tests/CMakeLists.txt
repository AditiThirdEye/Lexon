add_subdirectory(Interface)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LEXON_TEST_DEPENDS
  FileCheck count not
  lexon-opt
  lexon-translate
  lexon-container-test
  lexon-audio-container-test
  lexon-text-container-test
  mlir-runner
  )

if(LEXON_ENABLE_OPENCV)
  list(APPEND LEXON_TEST_DEPENDS lexon-image-container-test)
endif()

if(LEXON_MLIR_ENABLE_DIP_LIB)
  list(APPEND LEXON_TEST_DEPENDS lexon-new-image-container-test-bmp)
  if(LEXON_ENABLE_PNG)
    list(APPEND LEXON_TEST_DEPENDS lexon-new-image-container-test-png)
  endif()
endif()

if(LEXON_MLIR_ENABLE_PYTHON_PACKAGES)
  list(APPEND LEXON_TEST_DEPENDS LexonMLIRPythonModules)
endif()

add_lit_testsuite(check-tests "Running the lexon regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LEXON_TEST_DEPENDS}
  )
set_target_properties(check-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(LEXON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${LEXON_TEST_DEPENDS})

#------------------------------------------------------------------------------------
#    E2E test module
#------------------------------------------------------------------------------------

option(LEXON_ENABLE_E2E_TESTS "Enable end-to-end (E2E) tests for Models" OFF)

if(LEXON_ENABLE_E2E_TESTS)

  # Always enable PNG and DIP lib support
  set(LEXON_ENABLE_PNG ON CACHE BOOL "Enable PNG support" FORCE)
  set(LEXON_MLIR_ENABLE_DIP_LIB ON CACHE BOOL "Enable DIP library" FORCE)

  add_subdirectory(Models)
  set(LEXON_MODELS_TEST_DEPENDS
  FileCheck count not
  lexon-opt
  lexon-translate
  mlir-runner
  lexon-lenet-run-test-cpu
  )

  add_lit_testsuite(check-e2e "Running E2E tests for Models directory..."
    ${CMAKE_CURRENT_BINARY_DIR}/Models
    DEPENDS ${LEXON_MODELS_TEST_DEPENDS}
    )
  set_target_properties(check-e2e PROPERTIES FOLDER "Tests")

  add_lit_testsuites(LEXON_MODELS ${CMAKE_CURRENT_SOURCE_DIR}/Models DEPENDS ${LEXON_MODELS_TEST_DEPENDS})

endif()
