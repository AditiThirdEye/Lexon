//====- lexon-opt.cpp - The driver of lexon-mlir --------------------------===//
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
//
// This file is the dialects and optimization driver of lexon-mlir project.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"
#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"
#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "GPU/TransformOps.h"
#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "RVV/RVVDialect.h"
#include "VIR/VIRAttrs.h"
#include "VIR/VIRDialect.h"
#include "VIR/VIROps.h"
#include "VIR/VIRTypes.h"
#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"

namespace mlir {
namespace lexon {
void registerConvVectorizationPass();
void registerPointwiseConvToGemmPass();
void registerPointwiseConvToGemmForNhwcFhwcPass();
void registerPoolingVectorizationPass();
void registerPoolingNhwcMaxVectorizationPass();
void registerLowerBudPass();
void registerLowerDIPPass();
void registerBatchMatMulOptimizePass();
void registerBatchMatMulTileOptimizePass();
void registerBatchMatMuSCFOptimize();
void registerBatchMatMulTransVecPass();
void registerBatchMatMulVectorizationDecodePass();
void registerLowerDAPPass();
void registerExtendDAPPass();
void registerDAPVectorizePass();
void registerLowerRVVPass();
void registerMatMulOptimizePass();
void registerMatMulVectorizationPass();
void registerMatMulVectorizationDecodePass();
void registerMatMulParallelVectorizationPass();
void registerMatMulTransposeBUnrollVecPass();
void registerMatmulAMXPass();
void registerTransposeOptimizationPass();
void registerConvOptimizePass();
void registerConvNhwcFhwcOptimizePass();
void registerConvNhwcFhwcTileOptimizePass();
void registerDepthwiseConv2DNhwcHwcOptimizePass();
void registerLowerVectorExpPass();
void registerLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
void registerFuncBufferizeDynamicOffsetPass();
void registerAssumeTightMemRefLayoutPass();
void registerConvertMemcpyToGPUPass();
void registerLegalizeShmemOutliningPass();
void registerMatMulTransposeBVecPass();
void registerLegalizeShmemOutliningPass();
void registerVIRToVectorPass();
void registerLinalgToVIRPass();
void registerMatMulVectorizationBLISPass();
void registerSimplifyTosaReshapePass();
void registerSiLUFusionPass();
void registerSimplifyTosaMatmulScalarPass();
} // namespace lexon
} // namespace mlir

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  mlir::lexon::registerPointwiseConvToGemmPass();
  // Register Vectorization of Convolution.
  mlir::lexon::registerConvVectorizationPass();
  // Register Vectorization of Pooling.
  mlir::lexon::registerPoolingVectorizationPass();
  // Register Vectorization of Pooling Nhwc Max.
  mlir::lexon::registerPoolingNhwcMaxVectorizationPass();
  mlir::lexon::registerLowerBudPass();
  mlir::lexon::registerLowerDIPPass();
  mlir::lexon::registerLowerDAPPass();
  mlir::lexon::registerExtendDAPPass();
  // Register Vectorization of DAP Dialect.
  mlir::lexon::registerDAPVectorizePass();
  mlir::lexon::registerLowerRVVPass();
  mlir::lexon::registerLowerVectorExpPass();
  mlir::lexon::registerLowerGemminiPass();
  mlir::lexon::registerLowerLinalgToGemminiPass();

  // Register Several Optimize Pass.
  mlir::lexon::registerMatMulVectorizationBLISPass();
  mlir::lexon::registerMatMulOptimizePass();
  mlir::lexon::registerBatchMatMulOptimizePass();
  mlir::lexon::registerBatchMatMulTileOptimizePass();
  mlir::lexon::registerBatchMatMuSCFOptimize();
  mlir::lexon::registerBatchMatMulTransVecPass();
  mlir::lexon::registerBatchMatMulVectorizationDecodePass();
  mlir::lexon::registerMatMulVectorizationPass();
  mlir::lexon::registerMatMulVectorizationDecodePass();
  mlir::lexon::registerMatMulParallelVectorizationPass();
  mlir::lexon::registerMatMulTransposeBUnrollVecPass();
  mlir::lexon::registerMatmulAMXPass();
  mlir::lexon::registerTransposeOptimizationPass();
  mlir::lexon::registerConvOptimizePass();
  mlir::lexon::registerConvNhwcFhwcOptimizePass();
  mlir::lexon::registerConvNhwcFhwcTileOptimizePass();
  mlir::lexon::registerDepthwiseConv2DNhwcHwcOptimizePass();
  mlir::lexon::registerFuncBufferizeDynamicOffsetPass();
  mlir::lexon::registerAssumeTightMemRefLayoutPass();
  mlir::lexon::registerMatMulTransposeBVecPass();
  mlir::lexon::registerVIRToVectorPass();
  mlir::lexon::registerLinalgToVIRPass();
  // Register minimal TOSA reshape simplification pass.
  mlir::lexon::registerSimplifyTosaReshapePass();
  // Register tosa.matmul(K=1,J=1) -> tosa.mul rewrite.
  mlir::lexon::registerSimplifyTosaMatmulScalarPass();
  mlir::lexon::registerSiLUFusionPass();
  // Register gpu passes
  mlir::lexon::registerConvertMemcpyToGPUPass();
  mlir::lexon::registerLegalizeShmemOutliningPass();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  // Register dialects in lexon-mlir project.
  // clang-format off
  registry.insert<lexon::bud::BudDialect,
                  lexon::dip::DIPDialect,
                  lexon::dap::DAPDialect,
                  lexon::rvv::RVVDialect,
                  lexon::vector_exp::VectorExpDialect,
                  lexon::vir::VIRDialect,
                  lexon::gemmini::GemminiDialect>();
  // clang-format on

  mlir::lexon::registerLexonGPUTransformOps(registry);

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "lexon-mlir optimizer driver", registry));
}
