#pragma once

#include <mlir/Transforms/DialectConversion.h>

#include "dialect/colgm/unary_op.hpp"

namespace colgm_mlir {

struct lowering_neg : public mlir::ConversionPattern {
    explicit lowering_neg(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, neg_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_relu : public mlir::ConversionPattern {
    explicit lowering_relu(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, relu_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_abs : public mlir::ConversionPattern {
    explicit lowering_abs(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, abs_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_exp : public mlir::ConversionPattern {
    explicit lowering_exp(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, exp_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_log : public mlir::ConversionPattern {
    explicit lowering_log(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, log_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_sqrt : public mlir::ConversionPattern {
    explicit lowering_sqrt(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, sqrt_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_tanh : public mlir::ConversionPattern {
    explicit lowering_tanh(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, tanh_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_sigmoid : public mlir::ConversionPattern {
    explicit lowering_sigmoid(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, sigmoid_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_sin : public mlir::ConversionPattern {
    explicit lowering_sin(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, sin_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_cos : public mlir::ConversionPattern {
    explicit lowering_cos(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cos_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_gelu : public mlir::ConversionPattern {
    explicit lowering_gelu(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, gelu_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

}
