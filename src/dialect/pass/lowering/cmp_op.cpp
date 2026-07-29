#include <mlir/Dialect/Arith/IR/Arith.h>

#include "dialect/pass/lowering/cmp_op.hpp"

namespace colgm_mlir {

static mlir::LogicalResult
lowering_cmp_impl(mlir::Operation* op,
                  mlir::arith::CmpIPredicate iPred,
                  mlir::arith::CmpFPredicate fPred,
                  mlir::ConversionPatternRewriter& rewriter) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto ty = llvm::cast<mlir::RankedTensorType>(lhs.getType());
    auto base_type = ty.getElementType();

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto new_op = mlir::arith::CmpIOp::create(
            rewriter, op->getLoc(), iPred, lhs, rhs
        );
        rewriter.replaceOp(op, new_op->getResults());
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto new_op = mlir::arith::CmpFOp::create(
            rewriter, op->getLoc(), fPred, lhs, rhs
        );
        rewriter.replaceOp(op, new_op->getResults());
    } else {
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult
lowering_cmp_eq::matchAndRewrite(mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> operands,
                                 mlir::ConversionPatternRewriter& rewriter) const {
    return lowering_cmp_impl(op,
        mlir::arith::CmpIPredicate::eq,
        mlir::arith::CmpFPredicate::OEQ,
        rewriter);
}

mlir::LogicalResult
lowering_cmp_ne::matchAndRewrite(mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> operands,
                                 mlir::ConversionPatternRewriter& rewriter) const {
    return lowering_cmp_impl(op,
        mlir::arith::CmpIPredicate::ne,
        mlir::arith::CmpFPredicate::ONE,
        rewriter
    );
}

mlir::LogicalResult
lowering_cmp_lt::matchAndRewrite(mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> operands,
                                 mlir::ConversionPatternRewriter& rewriter) const {
    return lowering_cmp_impl(op,
        mlir::arith::CmpIPredicate::slt,
        mlir::arith::CmpFPredicate::OLT,
        rewriter
    );
}

mlir::LogicalResult
lowering_cmp_le::matchAndRewrite(mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> operands,
                                 mlir::ConversionPatternRewriter& rewriter) const {
    return lowering_cmp_impl(op,
        mlir::arith::CmpIPredicate::sle,
        mlir::arith::CmpFPredicate::OLE,
        rewriter
    );
}

mlir::LogicalResult
lowering_cmp_gt::matchAndRewrite(mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> operands,
                                 mlir::ConversionPatternRewriter& rewriter) const {
    return lowering_cmp_impl(op,
        mlir::arith::CmpIPredicate::sgt,
        mlir::arith::CmpFPredicate::OGT,
        rewriter
    );
}

mlir::LogicalResult
lowering_cmp_ge::matchAndRewrite(mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> operands,
                                 mlir::ConversionPatternRewriter& rewriter) const {
    return lowering_cmp_impl(op,
        mlir::arith::CmpIPredicate::sge,
        mlir::arith::CmpFPredicate::OGE,
        rewriter
    );
}

}
