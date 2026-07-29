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

struct lowering_elements : public mlir::ConversionPattern {
    explicit lowering_elements(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, elements_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_stack : public mlir::ConversionPattern {
    explicit lowering_stack(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, stack_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_reshape : public mlir::ConversionPattern {
    explicit lowering_reshape(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, reshape_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_transpose : public mlir::ConversionPattern {
    explicit lowering_transpose(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, transpose_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_broadcast : public mlir::ConversionPattern {
    explicit lowering_broadcast(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, broadcast_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_reduce_sum : public mlir::ConversionPattern {
    explicit lowering_reduce_sum(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, reduce_sum::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_reduce_max : public mlir::ConversionPattern {
    explicit lowering_reduce_max(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, reduce_max::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_gather : public mlir::ConversionPattern {
    explicit lowering_gather(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, gather_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

struct lowering_print : public mlir::ConversionPattern {
    explicit lowering_print(const mlir::TypeConverter& cvt, mlir::MLIRContext* ctx)
        : mlir::ConversionPattern(cvt, print_op::getOperationName(), 1, ctx) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op,
                    llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter& rewriter) const override;
};

}
