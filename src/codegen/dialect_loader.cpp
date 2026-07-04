
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "dialect/dialect.hpp"
#include "codegen/dialect_loader.hpp"

namespace colgm_mlir {

void load_dialect(mlir::MLIRContext& ctx) {
    ctx.getOrLoadDialect<colgm_dialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx.getOrLoadDialect<mlir::math::MathDialect>();
    ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();
    ctx.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
}

}