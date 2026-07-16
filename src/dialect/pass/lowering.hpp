#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
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

struct lowering_cast : public mlir::ConversionPattern {
    explicit lowering_cast(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cast_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_slice : public mlir::ConversionPattern {
    explicit lowering_slice(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, slice_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_cmp_eq : public mlir::ConversionPattern {
    explicit lowering_cmp_eq(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cmp_eq_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_cmp_ne : public mlir::ConversionPattern {
    explicit lowering_cmp_ne(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cmp_ne_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_cmp_lt : public mlir::ConversionPattern {
    explicit lowering_cmp_lt(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cmp_lt_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_cmp_le : public mlir::ConversionPattern {
    explicit lowering_cmp_le(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cmp_le_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_cmp_gt : public mlir::ConversionPattern {
    explicit lowering_cmp_gt(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cmp_gt_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_cmp_ge : public mlir::ConversionPattern {
    explicit lowering_cmp_ge(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, cmp_ge_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

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

struct lowering_yield : public mlir::ConversionPattern {
    explicit lowering_yield(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, yield_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_if : public mlir::ConversionPattern {
    explicit lowering_if(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, if_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_for : public mlir::ConversionPattern {
    explicit lowering_for(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, for_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_matmul : public mlir::ConversionPattern {
    explicit lowering_matmul(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, matmul_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

}
