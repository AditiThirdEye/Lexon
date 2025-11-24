if (${LEXON_DIP_OPT_STRIP_MINING})
    set(SPLITING_SIZE ${LEXON_DIP_OPT_STRIP_MINING})
elseif (HAVE_AVX512)
    set(SPLITING_SIZE 64)
elseif (HAVE_AVX2)
    set(SPLITING_SIZE 16)
elseif (HAVE_SSE)
    set(SPLITING_SIZE 16)
elseif (HAVE_NEON)
    set(SPLITING_SIZE 16)
endif ()

#-------------------------------------------------------------------------------
# Generate Lexon DIP Library: LexonLibDIP
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT DIP.o
        COMMAND ${CMAKE_BINARY_DIR}/bin/lexon-opt ${CMAKE_CURRENT_SOURCE_DIR}/DIP.mlir
        -lower-dip="DIP-strip-mining=${SPLITING_SIZE}"
        -arith-expand
        -lower-affine
        -convert-scf-to-cf
        -convert-math-to-llvm
        -convert-vector-to-llvm
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -convert-cf-to-llvm
        -convert-arith-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llc
        -mtriple=${LEXON_TARGET_TRIPLE}
        -mattr=${LEXON_OPT_ATTR}
        --filetype=obj
        -o ${CMAKE_CURRENT_BINARY_DIR}/DIP.o
        DEPENDS mlir-translate llc lexon-opt
        )

add_library(LexonLibDIP STATIC DIP.o)

SET_TARGET_PROPERTIES(LexonLibDIP PROPERTIES
  LINKER_LANGUAGE C
  ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
  )

#-------------------------------------------------------------------------------
# Generate Lexon DAP Library: LexonLibDAP
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT DAP.o
  COMMAND
    ${CMAKE_BINARY_DIR}/bin/lexon-opt ${CMAKE_CURRENT_SOURCE_DIR}/DAP.mlir
      -lower-dap="DAP-vector-splitting=${SPLITING_SIZE}"
      --convert-linalg-to-affine-loops
      -arith-expand
      -lower-affine
      -convert-scf-to-cf
      -convert-math-to-llvm
      -convert-vector-to-llvm
      -finalize-memref-to-llvm
      -llvm-request-c-wrappers
      -convert-func-to-llvm
      -convert-cf-to-llvm
      -convert-arith-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_TOOLS_BINARY_DIR}/llc
      -mtriple=${LEXON_TARGET_TRIPLE}
      -mattr=${LEXON_OPT_ATTR}
      --filetype=obj
      -o ${CMAKE_CURRENT_BINARY_DIR}/DAP.o
  DEPENDS mlir-translate llc lexon-opt
)

add_custom_command(
  OUTPUT DAP-extend.o
  COMMAND
    ${CMAKE_BINARY_DIR}/bin/lexon-opt ${CMAKE_CURRENT_SOURCE_DIR}/DAP-extend.mlir
      -extend-dap
      -one-shot-bufferize
      -convert-linalg-to-loops
      -convert-scf-to-cf
      -expand-strided-metadata
      -lower-affine
      -convert-vector-to-llvm
      -memref-expand
      -arith-expand
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-math-to-llvm
      -llvm-request-c-wrappers
      -convert-func-to-llvm
      -convert-cf-to-llvm
      -convert-arith-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BINARY_DIR}/llc
      -mtriple=${LEXON_TARGET_TRIPLE}
      -mattr=${LEXON_OPT_ATTR}
      -filetype=obj -relocation-model=pic
      -o ${CMAKE_CURRENT_BINARY_DIR}/DAP-extend.o
  DEPENDS mlir-translate llc lexon-opt
)

add_custom_command(
  OUTPUT DAPVectorization.o
  COMMAND
    cat ${CMAKE_CURRENT_SOURCE_DIR}/DAP.mlir |
    sed -e 's/@lexon_fir/@lexon_fir_vectorization/'
        -e 's/@lexon_iir/@lexon_iir_vectorization/'
        -e 's/@lexon_biquad/@lexon_biquad_vectorization/' |
    ${CMAKE_BINARY_DIR}/bin/lexon-opt
      -vectorize-dap
      -convert-linalg-to-affine-loops
      -arith-expand
      -lower-affine
      -convert-scf-to-cf
      -convert-math-to-llvm
      -convert-vector-to-llvm
      -finalize-memref-to-llvm
      -llvm-request-c-wrappers
      -convert-func-to-llvm
      -convert-cf-to-llvm
      -convert-arith-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_TOOLS_BINARY_DIR}/llc
      -mtriple=${LEXON_TARGET_TRIPLE}
      -mattr=${LEXON_OPT_ATTR}
      -filetype=obj
      -o ${CMAKE_CURRENT_BINARY_DIR}/DAPVectorization.o
  DEPENDS mlir-translate llc lexon-opt
)

add_library(LexonLibDAP STATIC
  DAP.o
  DAP-extend.o
  DAPVectorization.o
)

SET_TARGET_PROPERTIES(LexonLibDAP PROPERTIES
  LINKER_LANGUAGE CXX
  ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
)
