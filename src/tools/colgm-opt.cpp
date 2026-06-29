#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "dialect/pass/constfold.hpp"
#include "dialect/dialect.hpp"
#include "dialect/pass/colgm_lowering.hpp"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<colgm_mlir::colgm_dialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::scf::SCFDialect>();

    // Force-link constfold pass (static registration via PassRegistration
    // would be dropped by linker since constfold.cpp is in a .a archive).
    (void)colgm_mlir::create_colgm_const_fold_pass();
    (void)colgm_mlir::create_colgm_lowering_pass();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "colgm-opt", registry));
}
