add_mlir_library(LexonUtils
  Utils.cpp
  DIPUtils.cpp
  DAPUtils.cpp
  GPUUtils.cpp
  AffineTransformUtils.cpp
  )

add_mlir_library(LexonDIPUtils
  DIPUtils.cpp

  LINK_LIBS PUBLIC
  LexonUtils
  )

add_mlir_library(LexonDAPUtils
  DAPUtils.cpp
  
  LINK_LIBS PUBLIC
  LexonUtils
  )

add_mlir_library(LexonGPUUtils
  GPUUtils.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMTargetParser
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRAnalysis
  MLIRArithDialect
  MLIRArithUtils
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRViewLikeInterface
  MLIRGPUPasses
)
