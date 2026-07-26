#pragma once

#include "codegen/generator.hpp"

#include <mlir/IR/MLIRContext.h>

namespace colgm_mlir {

void run_jit(mlir::MLIRContext&, colgm_mlir::codegen&);

}
