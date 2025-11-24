//===------------- InitAll.h - Register all dialects and passes -----------===//
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

#ifndef LEXON_MLIR_INITALL_H
#define LEXON_MLIR_INITALL_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace lexon {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace lexon
} // namespace mlir

#endif // LEXON_MLIR_INITALL_H
