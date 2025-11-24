add_mlir_dialect_library(LexonDIP
  DIPDialect.cpp
  DIPOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )
