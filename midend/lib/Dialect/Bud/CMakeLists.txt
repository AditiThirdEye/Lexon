add_mlir_dialect_library(LexonBud
  BudDialect.cpp
  BudOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
