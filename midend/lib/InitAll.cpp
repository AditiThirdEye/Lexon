//===----------- InitAll.cpp - Register all dialects and passes -----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "lexon-mlir-c/InitAll.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"

#include "Dialect/Bud/BudDialect.h"
#include "Dialect/DAP/DAPDialect.h"
#include "Dialect/DIP/DIPDialect.h"
#include "Dialect/Gemmini/GemminiDialect.h"
#include "Dialect/RVV/RVVDialect.h"
#include "Dialect/VectorExp/VectorExpDialect.h"

namespace mlir {
namespace lexon {
void registerConvOptimizePass();
void registerConvVectorizationPass();
void registerPointwiseConvToGemmPass();
void registerPoolingVectorizationPass();
void registerLowerBudPass();
void registerLowerDAPPass();
void registerLowerDIPPass();
void registerLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
void registerLowerRVVPass();
void registerLowerVectorExpPass();
void registerBatchMatMulOptimizePass();
void registerMatMulOptimizePass();
void registerMatMulParallelVectorizationPass();
void registerMatMulVectorizationPass();
void registerMatMulVectorizationDecodePass();
void registerBatchMatMulVectorizationDecodePass();
void registerTransposeOptimizationPass();
void registerSiLUFusionPass();
void registerAssumeTightMemRefLayoutPass();
void registerSimplifyTosaReshapePass();
void registerSimplifyTosaMatmulScalarPass();
} // namespace lexon
} // namespace mlir

void mlir::lexon::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<::lexon::bud::BudDialect>();
  registry.insert<::lexon::dap::DAPDialect>();
  registry.insert<::lexon::dip::DIPDialect>();
  registry.insert<::lexon::gemmini::GemminiDialect>();
  registry.insert<::lexon::rvv::RVVDialect>();
  registry.insert<::lexon::vector_exp::VectorExpDialect>();
}

void mlir::lexon::registerAllPasses() {
  mlir::lexon::registerConvOptimizePass();
  mlir::lexon::registerConvVectorizationPass();
  mlir::lexon::registerPointwiseConvToGemmPass();
  mlir::lexon::registerPoolingVectorizationPass();
  mlir::lexon::registerLowerBudPass();
  mlir::lexon::registerLowerDAPPass();
  mlir::lexon::registerLowerDIPPass();
  mlir::lexon::registerLowerGemminiPass();
  mlir::lexon::registerLowerLinalgToGemminiPass();
  mlir::lexon::registerLowerRVVPass();
  mlir::lexon::registerLowerVectorExpPass();
  mlir::lexon::registerBatchMatMulOptimizePass();
  mlir::lexon::registerMatMulOptimizePass();
  mlir::lexon::registerMatMulParallelVectorizationPass();
  mlir::lexon::registerMatMulVectorizationPass();
  mlir::lexon::registerMatMulVectorizationDecodePass();
  mlir::lexon::registerBatchMatMulVectorizationDecodePass();
  mlir::lexon::registerTransposeOptimizationPass();
  mlir::lexon::registerSiLUFusionPass();
  mlir::lexon::registerAssumeTightMemRefLayoutPass();
  mlir::lexon::registerSimplifyTosaReshapePass();
  mlir::lexon::registerSimplifyTosaMatmulScalarPass();
}
