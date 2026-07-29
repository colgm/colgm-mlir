#include <mlir/Dialect/Arith/IR/Arith.h>

#include "dialect/pass/lowering/binary_op.hpp"

namespace colgm_mlir {

mlir::LogicalResult
lowering_add::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto add = llvm::cast<add_op>(op);
    auto lhs = add.get_lhs();
    auto rhs = add.get_rhs();
    mlir::ValueRange vr = { lhs, rhs };

    auto ty = llvm::cast<mlir::RankedTensorType>(lhs.getType());
    auto base_type = ty.getElementType();

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto new_op = mlir::arith::AddIOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto new_op = mlir::arith::AddFOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else {
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult
lowering_sub::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto sub = llvm::cast<sub_op>(op);
    auto lhs = sub.get_lhs();
    auto rhs = sub.get_rhs();
    mlir::ValueRange vr = { lhs, rhs };

    auto ty = llvm::cast<mlir::RankedTensorType>(lhs.getType());
    auto base_type = ty.getElementType();

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto new_op = mlir::arith::SubIOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto new_op = mlir::arith::SubFOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else {
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult
lowering_mul::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto mul = llvm::cast<mul_op>(op);
    auto lhs = mul.get_lhs();
    auto rhs = mul.get_rhs();
    mlir::ValueRange vr = { lhs, rhs };

    auto ty = llvm::cast<mlir::RankedTensorType>(lhs.getType());
    auto base_type = ty.getElementType();

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto new_op = mlir::arith::MulIOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto new_op = mlir::arith::MulFOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else {
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult
lowering_div::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto div = llvm::cast<div_op>(op);
    auto lhs = div.get_lhs();
    auto rhs = div.get_rhs();
    mlir::ValueRange vr = { lhs, rhs };

    auto ty = llvm::cast<mlir::RankedTensorType>(lhs.getType());
    auto base_type = ty.getElementType();

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto new_op = mlir::arith::DivSIOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto new_op = mlir::arith::DivFOp::create(rewriter, op->getLoc(), vr);
        rewriter.replaceOp(op, new_op->getResults());
    } else {
        return mlir::failure();
    }
    return mlir::success();
}

}
