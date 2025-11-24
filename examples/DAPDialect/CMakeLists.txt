if (${LEXON_DAP_OPT_VECTOR_SPLITTING})
  set(SPLITING_SIZE ${LEXON_DAP_OPT_VECTOR_SPLITTING})
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256)
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128)
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64)
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64)
endif()

message(STATUS "Spliting size: ${SPLITING_SIZE}")

#-------------------------------------------------------------------------------
# Lexon DAP Dialect FIR operation
#-------------------------------------------------------------------------------

add_executable(lexon-fir-scalar FIRLowpass.cpp)
add_dependencies(lexon-fir-scalar lexon-opt)
target_link_libraries(lexon-fir-scalar
  LexonLibDAP
  mlir_c_runner_utils
)

add_executable(lexon-fir-vectorization FIRVectorization.cpp)
add_dependencies(lexon-fir-vectorization lexon-opt)
target_link_libraries(lexon-fir-vectorization
  LexonLibDAP
  mlir_c_runner_utils
)

#-------------------------------------------------------------------------------
# Lexon DAP Dialect Biquad Operation
#-------------------------------------------------------------------------------

add_executable(lexon-biquad biquad.cpp)
add_dependencies(lexon-biquad lexon-opt)
target_link_libraries(lexon-biquad
  LexonLibDAP
  mlir_c_runner_utils
)

#-------------------------------------------------------------------------------
# Lexon DAP Dialect IIR Operation
#-------------------------------------------------------------------------------

add_executable(lexon-iir-scalar IIRLowpass.cpp)
add_dependencies(lexon-iir-scalar lexon-opt)
target_link_libraries(lexon-iir-scalar 
  LexonLibDAP
  mlir_c_runner_utils
)

add_executable(lexon-iir-vectorization IIRVectorization.cpp)
add_dependencies(lexon-iir-vectorization lexon-opt)
target_link_libraries(lexon-iir-vectorization
  LexonLibDAP
  mlir_c_runner_utils
)

#-------------------------------------------------------------------------------
# Lexon DAP Dialect WhisperPreprocess Operation
#-------------------------------------------------------------------------------

add_executable(lexon-whisper-preprocess WhisperPreprocess.cpp)
add_dependencies(lexon-whisper-preprocess lexon-opt)
target_link_libraries(lexon-whisper-preprocess
  LexonLibDAP
  mlir_c_runner_utils
)

add_executable(lexon-rfft RFFT.cpp)
add_dependencies(lexon-rfft lexon-opt)
target_link_libraries(lexon-rfft
  LexonLibDAP
  mlir_c_runner_utils
)
