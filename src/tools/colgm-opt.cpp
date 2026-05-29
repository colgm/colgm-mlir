#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "dialect/dialect.hpp"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<colgm_mlir::colgm_dialect>();
    registry.insert<mlir::func::FuncDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "colgm-opt", registry));
}
