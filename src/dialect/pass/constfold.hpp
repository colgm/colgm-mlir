#pragma once

#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <memory>

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> create_colgm_const_fold_pass();

class colgm_const_fold_pass: public mlir::PassWrapper<colgm_const_fold_pass,
                                                      mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(colgm_const_fold_pass)

    llvm::StringRef getArgument() const override { return "colgm-const-fold"; }
    llvm::StringRef getDescription() const override {
        return "Fold colgm.elements with all-constant operands into colgm.constant";
    }
    void runOnOperation() override;
};

}

