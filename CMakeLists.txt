##===- CMakeLists.txt - lexon-mlir cmake root -----------------*- cmake -*-===//
##
## Configure the lexon-mlir build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

if(POLICY CMP0174)
  cmake_policy(SET CMP0174 NEW)
endif()
#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(lexon-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
include(ExternalProject)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)
option(LEXON_MLIR_OUT_OF_TREE_BUILD "Specifies an out of tree build" OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR LEXON_MLIR_OUT_OF_TREE_BUILD)
    message(STATUS "lexon-mlir two-step build.")
    # Two-step build
    #-------------------------------------------------------------------------------
    # MLIR/LLVM Configuration
    #-------------------------------------------------------------------------------
    find_package(MLIR REQUIRED CONFIG)
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    # LLVM_MAIN_SRC_DIR is a private variable for the LLVM in-tree build.
    # To provide compatibility for unifying the one-step and two-step build,
    # we set LLVM_MAIN_SRC_DIR ourselves here.
    # This could benefit users who want to specify a custom LLVM source directory,
    # but also not interfere with normal users who just want to use the lexon-mlir provided LLVM sources.
    if(NOT DEFINED LLVM_MAIN_SRC_DIR)
      get_filename_component(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm/llvm ABSOLUTE)
    endif()
    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

    find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
            NO_DEFAULT_PATH)

    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
else()
    message(STATUS "lexon-mlir one-step build.")
    # one-step build with LLVM_EXTERNAL_PROJECTS=lexon-mlir
    #-------------------------------------------------------------------------------
    # MLIR/LLVM Configuration
    #-------------------------------------------------------------------------------
    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
    set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
    set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
endif()

#-------------------------------------------------------------------------------
# LEXON configuration
#-------------------------------------------------------------------------------

# LEXON project.
set(LEXON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LEXON_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LEXON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LEXON_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(LEXON_EXAMPLES_DIR ${LEXON_SOURCE_DIR}/examples)
set(LEXON_MIDEND_INCLUDE_DIR ${LEXON_SOURCE_DIR}/midend/include)
set(LEXON_THIRDPARTY_INCLUDE_DIR ${LEXON_SOURCE_DIR}/thirdparty/include)
set(LEXON_MLIR_PYTHON_PACKAGES_DIR ${LEXON_BUILD_DIR}/python_packages)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LEXON_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LEXON_LIBRARY_DIR})

set(LEXON_EXAMPLES OFF CACHE BOOL "Build examples")
set(LEXON_TRANSFORMER_EXAMPLES OFF CACHE BOOL "Build transformer examples")
set(LEXON_ENABLE_OPENCV OFF CACHE BOOL "Enable OpenCV support.")

 if(LEXON_ENABLE_OPENCV)
   add_definitions(-DLEXON_ENABLE_OPENCV)
   find_package(JPEG REQUIRED)
   find_package(PNG REQUIRED)
   find_package(OpenCV REQUIRED CONFIG)
   include_directories(${OpenCV_INCLUDE_DIRS})
 endif()

if(LEXON_MLIR_ENABLE_DIP_LIB)
  add_definitions(-DLEXON_MLIR_ENABLE_DIP_LIB)
  find_package(PNG REQUIRED)
endif()

if(LEXON_ENABLE_PNG)
  add_definitions(-DLEXON_ENABLE_PNG)
  find_package(PNG REQUIRED)
endif()

# Generate libraries into `lib` of build directory.
set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Add LEXON files to the include path
include_directories(${LEXON_MAIN_INCLUDE_DIR})
include_directories(${LEXON_MIDEND_INCLUDE_DIR})
include_directories(${LEXON_MIDEND_INCLUDE_DIR}/Interface)
include_directories(${LEXON_MIDEND_INCLUDE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/midend/include/Dialect)
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/backend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LEXON_SOURCE_DIR}/lib)
include_directories(${LEXON_THIRDPARTY_INCLUDE_DIR})
include_directories(${LEXON_SOURCE_DIR}/frontend/Interfaces)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Configure CMake.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#-------------------------------------------------------------------------------
# DIP lib configuration
#-------------------------------------------------------------------------------

if(LEXON_MLIR_ENABLE_DIP_LIB)
  find_package(JPEG REQUIRED)
  find_package(PNG REQUIRED)
endif()

#-------------------------------------------------------------------------------
# Hardware detection
#-------------------------------------------------------------------------------

include(${LEXON_SOURCE_DIR}/cmake/check_simd.cmake)
include(${LEXON_SOURCE_DIR}/cmake/check_toolchain.cmake)
check_simd()
check_toolchain()

#-------------------------------------------------------------------------------
# Antlr Configuration
#-------------------------------------------------------------------------------
# NB: currently, ANTLR is used in dsl examples only,
# however, there is a plan to use in the frontend,
# so it is kept in the top-level cmake
if(LEXON_DSL_EXAMPLES)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Antlr)

    # required if linking to static library
    add_definitions(-DANTLR4CPP_STATIC)

    # add external build for antlrcpp
    include(ExternalAntlr4Cpp)
    # add antrl4cpp artifacts to project environment
    include_directories(${ANTLR4_INCLUDE_DIRS})

    # set variable pointing to the antlr tool that supports C++
    # this is not required if the jar file can be found under PATH environment
    set(ANTLR_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/antlr/antlr-4.10.1-complete.jar)
    # add macros to generate ANTLR Cpp code from grammar
    find_package(ANTLR REQUIRED)
endif()

#-------------------------------------------------------------------------------
# The `mimalloc` Configuration
#-------------------------------------------------------------------------------

if(LEXON_MLIR_USE_MIMALLOC)
  if(MIMALLOC_BUILD_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${MIMALLOC_BUILD_DIR})
  endif()
  find_package(mimalloc REQUIRED)
endif()

#-------------------------------------------------------------------------------
# The RISC-V toolchain
#-------------------------------------------------------------------------------

if(LEXON_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN)
  set(RISCV_GNU_TOOLCHAIN_DIR "${LEXON_SOURCE_DIR}/thirdparty/riscv-gnu-toolchain")
  set(RISCV_GNU_TOOLCHAIN_INSTALL_DIR "${CMAKE_BINARY_DIR}/thirdparty/riscv-gnu-toolchain")
  ExternalProject_Add(
    riscv-gnu-toolchain
    SOURCE_DIR        ${RISCV_GNU_TOOLCHAIN_DIR}
    PREFIX            ${RISCV_GNU_TOOLCHAIN_INSTALL_DIR}
    CONFIGURE_COMMAND ${RISCV_GNU_TOOLCHAIN_DIR}/configure --prefix=${RISCV_GNU_TOOLCHAIN_INSTALL_DIR}
    BUILD_COMMAND     make clean && make linux build-qemu -j
    BUILD_IN_SOURCE   TRUE
    INSTALL_COMMAND   ""
  )
endif()

#-------------------------------------------------------------------------------
# Initialize Python packages
#-------------------------------------------------------------------------------
if(LEXON_MLIR_ENABLE_PYTHON_PACKAGES)
  # Find the Python interpreter and development components,
  # requiring a minimum version of 3.10
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
  # Create directories for the LEXON-MLIR Python packages
  file(MAKE_DIRECTORY ${LEXON_MLIR_PYTHON_PACKAGES_DIR}/lexon)
  file(MAKE_DIRECTORY ${LEXON_MLIR_PYTHON_PACKAGES_DIR}/lexon/compiler)
  # Create empty __init__.py files to make these directories Python packages
  file(WRITE ${LEXON_MLIR_PYTHON_PACKAGES_DIR}/lexon/__init__.py "")
  file(WRITE ${LEXON_MLIR_PYTHON_PACKAGES_DIR}/lexon/compiler/__init__.py "")

  install(DIRECTORY ${LEXON_MLIR_PYTHON_PACKAGES_DIR}/lexon DESTINATION python_packages)
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(cmake)
add_subdirectory(frontend)
add_subdirectory(midend)
add_subdirectory(backend)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(tests)
#-------------------------------------------------------------------------------
# Target check-lexon
#-------------------------------------------------------------------------------

add_custom_target(check-lexon
  DEPENDS check-examples check-tests
)

#-------------------------------------------------------------------------------
# Target install
#-------------------------------------------------------------------------------

# Install frontend interfaces into include directory, so that the downstream project can use them in distribution
install(DIRECTORY lexon/Core lexon/DAP lexon/DIP lexon/LLM
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lexon-mlir
  COMPONENT lexon-mlir-interfaces-headers
  FILES_MATCHING
  PATTERN "*.h"
  )
