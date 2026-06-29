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

void colgm_lowering::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<colgm_dialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();

    
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<lowering_constant>(cvt, &getContext());

    mlir::FrozenRewritePatternSet frozen(std::move(patterns));
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, frozen))) {
        signalPassFailure();
    }
}

static mlir::PassRegistration<colgm_lowering> pass;

}
