#include "dialect/pass/colgm_lowering.hpp"

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> create_colgm_lowering_pass() {
    return std::make_unique<colgm_lowering>();
}

mlir::LogicalResult
lowering_constant::matchAndRewrite(mlir::Operation* op,
                                   llvm::ArrayRef<mlir::Value> operands,
                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto constant = llvm::cast<constant_op>(op);
    auto value = constant.get_value();

    auto new_op = mlir::arith::ConstantOp::create(rewriter, op->getLoc(), value);
    rewriter.replaceOp(op, new_op->getResults());
    return mlir::success();
}

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

mlir::LogicalResult
lowering_cast::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto cast = llvm::cast<cast_op>(op);
    auto input = cast.get_input();
    auto src = input.getType();
    auto dst = cast->getResult(0).getType();
    auto loc = cast.getLoc();

    auto src_tensor = llvm::dyn_cast<mlir::RankedTensorType>(src);
    auto dst_tensor = llvm::dyn_cast<mlir::RankedTensorType>(dst);

    if (src_tensor && !dst_tensor &&
        src_tensor.getRank() == 0 && src_tensor.getElementType() == dst) {
        auto extract = mlir::tensor::ExtractOp::create(
            rewriter, cast.getLoc(), input, mlir::ValueRange {}
        );
        // auto new_op = rewriter.create<mlir::tensor::ExtractOp>(loc, input, mlir::ValueRange{});
        rewriter.replaceOp(op, extract->getResults());
        return mlir::success();
    }

    return mlir::failure();
}

void colgm_lowering::runOnOperation() {
    cvt.addConversion([](mlir::Type type) { return type; });

    getContext().getOrLoadDialect<colgm_dialect>();
    getContext().getOrLoadDialect<mlir::func::FuncDialect>();
    getContext().getOrLoadDialect<mlir::arith::ArithDialect>();
    getContext().getOrLoadDialect<mlir::math::MathDialect>();
    getContext().getOrLoadDialect<mlir::tensor::TensorDialect>();
    getContext().getOrLoadDialect<mlir::linalg::LinalgDialect>();
    getContext().getOrLoadDialect<mlir::scf::SCFDialect>();

    mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<colgm_dialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<lowering_constant,
                  lowering_add, lowering_sub,
                  lowering_mul, lowering_div,
                  lowering_cast>(cvt, &getContext());

    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    (void)mlir::applyPartialConversion(getOperation(), target, frozen);
    // if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, frozen))) {
    //     signalPassFailure();
    // }
}

static mlir::PassRegistration<colgm_lowering> pass;

}
