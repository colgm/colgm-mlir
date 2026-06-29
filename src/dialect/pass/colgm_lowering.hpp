#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <memory>

#include "dialect/dialect.hpp"

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> create_colgm_lowering_pass();

class colgm_lowering : public mlir::PassWrapper<colgm_lowering,
                                               mlir::OperationPass<mlir::ModuleOp>> { 
private:
    mlir::TypeConverter cvt;

public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(colgm_lowering)

    llvm::StringRef getArgument() const override { return "colgm-lowering"; }
    llvm::StringRef getDescription() const override {
        return "Lowering colgm dialect to standard MLIR dialects";
    }
    void runOnOperation() override;
};

struct lowering_constant : public mlir::ConversionPattern {
    explicit lowering_constant(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, constant_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

}
