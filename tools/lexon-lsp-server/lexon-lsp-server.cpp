//====- lexon-lsp-server.cpp - The LSP server of lexon-mlir --------------------------===//
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
// This file is a LSP server for new dialects of lexon-mlir project.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"
#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"
#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "RVV/RVVDialect.h"
#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"
#include "Gemmini/GemminiDialect.h"

using namespace mlir;

#ifdef MLIR_INCLUDE_TESTS
namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
} // namespace test
#endif

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  // clang-format off
  registry.insert<lexon::bud::BudDialect,
                  lexon::dip::DIPDialect,
                  lexon::dap::DAPDialect,
                  lexon::rvv::RVVDialect,
                  lexon::vector_exp::VectorExpDialect,
                  lexon::gemmini::GemminiDialect>();
  // clang-format on
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
#endif
  return failed(MlirLspServerMain(argc, argv, registry));
}
