#include "codegen/dialect_loader.hpp"
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

static mlir::Value arith_cast(mlir::ConversionPatternRewriter& rewriter,
                              mlir::Location loc,
                              mlir::Value val, mlir::Type src, mlir::Type dst) {
    if (src == dst) {
        return val;
    }

    // int -> int: extsi (widen) / trunci (narrow)
    if (llvm::isa<mlir::IntegerType>(src) && llvm::isa<mlir::IntegerType>(dst)) {
        auto src_w = llvm::cast<mlir::IntegerType>(src).getWidth();
        auto dst_w = llvm::cast<mlir::IntegerType>(dst).getWidth();
        if (src_w < dst_w) {
            auto new_op = mlir::arith::ExtSIOp::create(rewriter, loc, dst, val);
            return new_op->getResult(0);
        } else if (src_w > dst_w) {
            auto new_op = mlir::arith::TruncIOp::create(rewriter, loc, dst, val);
            return new_op->getResult(0);
        }
    }

    // int -> float
    if (llvm::isa<mlir::IntegerType>(src) && llvm::isa<mlir::FloatType>(dst)) {
        auto new_op = mlir::arith::SIToFPOp::create(rewriter, loc, dst, val);
        return new_op->getResult(0);
    }

    // float -> int
    if (llvm::isa<mlir::FloatType>(src) && llvm::isa<mlir::IntegerType>(dst)) {
        auto new_op = mlir::arith::FPToSIOp::create(rewriter, loc, dst, val);
        return new_op->getResult(0);
    }

    // float -> float: extf / truncf
    if (llvm::isa<mlir::FloatType>(src) && llvm::isa<mlir::FloatType>(dst)) {
        auto src_w = llvm::cast<mlir::FloatType>(src).getWidth();
        auto dst_w = llvm::cast<mlir::FloatType>(dst).getWidth();
        if (src_w < dst_w) {
            auto new_op = mlir::arith::ExtFOp::create(rewriter, loc, dst, val);
            return new_op->getResult(0);
        } else {
            auto new_op = mlir::arith::TruncFOp::create(rewriter, loc, dst, val);
            return new_op->getResult(0);
        }
    }

    // int <-> index
    if ((llvm::isa<mlir::IntegerType>(src) && llvm::isa<mlir::IndexType>(dst)) ||
        (llvm::isa<mlir::IndexType>(src) && llvm::isa<mlir::IntegerType>(dst))) {
        auto new_op = mlir::arith::IndexCastOp::create(rewriter, loc, dst, val);
        return new_op->getResult(0);
    }

    return {};
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

    auto extract_scalar = [&](mlir::Value tensor) -> mlir::Value {
        auto elem_ty = llvm::cast<mlir::RankedTensorType>(tensor.getType()).getElementType();
        auto extract = mlir::tensor::ExtractOp::create(
            rewriter, loc, tensor, mlir::ValueRange {}
        );
        return extract->getResult(0);
    };

    // tensor<ty> -> ty
    if (src_tensor && !dst_tensor &&
        src_tensor.getRank() == 0 && src_tensor.getElementType() == dst) {
        auto extract = mlir::tensor::ExtractOp::create(
            rewriter, loc, input, mlir::ValueRange {}
        );
        rewriter.replaceOp(op, extract->getResults());
        return mlir::success();
    }

    // ty -> tensor<ty>
    if (!src_tensor && dst_tensor &&
        dst_tensor.getRank() == 0 && src == dst_tensor.getElementType()) {
        auto fe = mlir::tensor::FromElementsOp::create(
            rewriter, loc, dst, mlir::ValueRange { input }
        );
        rewriter.replaceOp(op, fe->getResults());
        return mlir::success();
    }

    // tensor<ty0> -> ty1 (maybe not used)
    if (src_tensor && !dst_tensor && src_tensor.getRank() == 0) {
        auto val = extract_scalar(input);
        auto casted = arith_cast(rewriter, loc, val,
                                 src_tensor.getElementType(), dst);
        rewriter.replaceOp(op, mlir::ValueRange { casted });
        return mlir::success();
    }

    // ty0 -> tensor<ty1> (maybe not used)
    if (!src_tensor && dst_tensor && dst_tensor.getRank() == 0) {
        auto val = arith_cast(rewriter, loc, input, src, dst_tensor.getElementType());
        if (!val) {
            return mlir::failure();
        }
        auto fe = mlir::tensor::FromElementsOp::create(
            rewriter, loc, dst, mlir::ValueRange { val }
        );
        rewriter.replaceOp(op, fe->getResults());
        return mlir::success();
    }

    // tensor<ty0> -> tensor<ty1>
    if (src_tensor && dst_tensor &&
        src_tensor.getRank() == 0 && dst_tensor.getRank() == 0) {
        auto val = extract_scalar(input);
        auto casted = arith_cast(
            rewriter, loc, val,
            src_tensor.getElementType(),
            dst_tensor.getElementType()
        );
        if (!casted) {
            return mlir::failure();
        }
        auto fe = mlir::tensor::FromElementsOp::create(
            rewriter, loc, dst, mlir::ValueRange { casted }
        );
        rewriter.replaceOp(op, fe->getResults());
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult
lowering_slice::matchAndRewrite(mlir::Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                mlir::ConversionPatternRewriter& rewriter) const {
    auto slice = llvm::cast<slice_op>(op);
    auto input = slice.get_input();
    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto index = slice.get_index();
    auto axis = slice.get_axis();
    auto result_type = llvm::cast<mlir::RankedTensorType>(slice->getResult(0).getType());

    auto rank = input_type.getRank();
    llvm::SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult> sizes(rank);
    llvm::SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    offsets[axis] = index;
    for (i64 i = 0; i < rank; ++i) {
        sizes[i] = rewriter.getIndexAttr(input_type.getDimSize(i));
    }
    sizes[axis] = rewriter.getIndexAttr(1);

    auto extract = mlir::tensor::ExtractSliceOp::create(
        rewriter, slice.getLoc(), result_type, input, offsets, sizes, strides
    );
    rewriter.replaceOp(op, extract->getResults());
    return mlir::success();
}

void colgm_lowering::runOnOperation() {
    cvt.addConversion([](mlir::Type type) { return type; });

    load_dialect(getContext());

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
                  lowering_cast, lowering_slice>(cvt, &getContext());

    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    (void)mlir::applyPartialConversion(getOperation(), target, frozen);
    // if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, frozen))) {
    //     signalPassFailure();
    // }
}

static mlir::PassRegistration<colgm_lowering> pass;

}
