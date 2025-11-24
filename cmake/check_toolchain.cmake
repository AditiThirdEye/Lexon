macro(CHECK_TOOLCHAIN)
#-------------------------------------------------------------------------------
# Check Triple
#-------------------------------------------------------------------------------
  set(LEXON_TARGET_TRIPLE "${LLVM_HOST_TRIPLE}" CACHE STRING "Target triple of the host machine")
  message(STATUS "Target Triple: ${LEXON_TARGET_TRIPLE}")

#-------------------------------------------------------------------------------
# Check Attribute
#-------------------------------------------------------------------------------
  set(LEXON_OPT_ATTR "" CACHE STRING "Target attribute of the host machine")  
  if ("${LEXON_OPT_ATTR}" STREQUAL "")
    if (HAVE_AVX512)
      # TODO: Figure out the difference of sse/sse2/sse4.1
      set(LEXON_OPT_ATTR avx512f)
    elseif(HAVE_AVX2)
      set(LEXON_OPT_ATTR avx2)
    elseif(HAVE_SSE)
      set(LEXON_OPT_ATTR sse)
    elseif(HAVE_NEON)
      set(LEXON_OPT_ATTR neon)
    endif()
  endif()
  message(STATUS "Target Attr: ${LEXON_OPT_ATTR}")

endmacro(CHECK_TOOLCHAIN)
