#pragma once

#include <mlir/Transforms/DialectConversion.h>

#include "dialect/colgm/binary_op.hpp"

namespace colgm_mlir {

struct lowering_add : public mlir::ConversionPattern {
    explicit lowering_add(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, add_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_sub : public mlir::ConversionPattern {
    explicit lowering_sub(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, sub_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_mul : public mlir::ConversionPattern {
    explicit lowering_mul(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, mul_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_div : public mlir::ConversionPattern {
    explicit lowering_div(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, div_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

}
