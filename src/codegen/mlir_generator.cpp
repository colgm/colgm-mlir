#include "codegen/mlir_generator.hpp"

namespace colgm_mlir {

void mlir_generator::generate(root* r) {
    module = mlir::ModuleOp::create(builder, to_loc(r));
    builder.setInsertionPointToStart(module.getBody());

    auto func = mlir::func::FuncOp::create(
        builder, builder.getUnknownLoc(), "main",
        builder.getFunctionType({}, {}));
    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
}

}
