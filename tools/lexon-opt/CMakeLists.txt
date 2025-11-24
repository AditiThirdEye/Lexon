get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_tool(lexon-opt
  lexon-opt.cpp
)

target_link_libraries(lexon-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  CBConvVectorization
  LexonBud
  LowerBudPass
  LexonDIP
  LowerDIPPass
  LexonDAP
  LowerDAPPass
  ExtendDAPPass
  DAPVectorization
  LexonRVV
  LowerRVVPass
  MatMulOptimization
  BatchMatMulOptimization
  MatMulParallelVectorization
  TransposeOptimization
  GraphRedundancyElimination
  ConvOptimization
  DepthwiseConvOptimization
  VectorExp
  LowerVectorExpPass
  LexonGemmini
  LowerGemminiPass
  LowerLinalgToGemminiPass
  FuncBufferizeDynamicOffset
  MLIRGPUPasses
  LexonGPUTransformOPs
  MLIRTestTransforms
  MLIRTestTransformDialect
  MLIRTransforms
  MLIRTransformUtils
  MatMulTransposeBVec
  LinalgToVIRPass
  VIRToVectorPass
  TosaElementwiseFusion
  AssumeTightMemRefLayoutPass
  )
