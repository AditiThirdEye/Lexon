add_mlir_library(TosaElementwiseFusion
  SiLUFusion.cpp

  LINK_LIBS PUBLIC
  LexonUtils
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRTransforms
)
