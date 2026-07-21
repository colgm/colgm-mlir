#include "codegen/dialect_loader.hpp"
#include "dialect/pass/lowering.hpp"

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

mlir::LogicalResult
lowering_neg::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto neg = llvm::cast<neg_op>(op);
    auto input = neg.get_input();
    auto ty = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto base_type = ty.getElementType();

    mlir::TypedAttr zero_attr;
    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto it = llvm::cast<mlir::IntegerType>(base_type);
        zero_attr = mlir::DenseElementsAttr::get(
            ty, llvm::APInt(it.getWidth(), 0)
        );
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto ft = llvm::cast<mlir::FloatType>(base_type);
        zero_attr = mlir::DenseElementsAttr::get(
            ty, llvm::APFloat(ft.getFloatSemantics(), "0")
        );
    } else {
        return mlir::failure();
    }

    auto zero = mlir::arith::ConstantOp::create(rewriter, op->getLoc(), zero_attr);

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto sub = mlir::arith::SubIOp::create(
            rewriter, op->getLoc(), mlir::ValueRange{ zero, input }
        );
        rewriter.replaceOp(op, sub->getResults());
    } else {
        auto sub = mlir::arith::SubFOp::create(
            rewriter, op->getLoc(), mlir::ValueRange{ zero, input }
        );
        rewriter.replaceOp(op, sub->getResults());
    }
    return mlir::success();
}

mlir::LogicalResult
lowering_relu::matchAndRewrite(mlir::Operation* op,
                               llvm::ArrayRef<mlir::Value> operands,
                               mlir::ConversionPatternRewriter& rewriter) const {
    auto relu = llvm::cast<relu_op>(op);
    auto input = relu.get_input();
    auto ty = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto base_type = ty.getElementType();

    mlir::TypedAttr zero_attr;
    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto it = llvm::cast<mlir::IntegerType>(base_type);
        zero_attr = mlir::DenseElementsAttr::get(
            ty, llvm::APInt(it.getWidth(), 0)
        );
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto ft = llvm::cast<mlir::FloatType>(base_type);
        zero_attr = mlir::DenseElementsAttr::get(
            ty, llvm::APFloat(ft.getFloatSemantics(), "0")
        );
    } else {
        return mlir::failure();
    }

    auto zero = mlir::arith::ConstantOp::create(rewriter, op->getLoc(), zero_attr);

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto new_max = mlir::arith::MaxSIOp::create(
            rewriter, op->getLoc(), mlir::ValueRange{ zero, input }
        );
        rewriter.replaceOp(op, new_max->getResults());
    } else {
        auto new_max = mlir::arith::MaximumFOp::create(
            rewriter, op->getLoc(), mlir::ValueRange{ zero, input }
        );
        rewriter.replaceOp(op, new_max->getResults());
    }
    return mlir::success();
}

mlir::LogicalResult
lowering_abs::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto abs = llvm::cast<abs_op>(op);
    auto input = abs.get_input();
    auto loc = abs.getLoc();
    auto ty = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto base_type = ty.getElementType();

    if (llvm::isa<mlir::IntegerType>(base_type)) {
        auto new_op = mlir::math::AbsIOp::create(rewriter, loc, input);
        rewriter.replaceOp(op, new_op->getResults());
    } else if (llvm::isa<mlir::FloatType>(base_type)) {
        auto fastmath = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::none
        );
        auto new_op = mlir::math::AbsFOp::create(rewriter, loc, input, fastmath);
        rewriter.replaceOp(op, new_op->getResults());
    } else {
        return mlir::failure();
    }
    return mlir::success();
}

static mlir::LogicalResult
lowering_math_unary(mlir::Operation* op,
                    mlir::Value input,
                    mlir::ConversionPatternRewriter& rewriter,
                    llvm::function_ref<mlir::Operation*(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::arith::FastMathFlagsAttr)> createFn) {
    auto ty = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto base_type = ty.getElementType();

    if (!llvm::isa<mlir::FloatType>(base_type)) {
        return mlir::failure();
    }

    auto fastmath = mlir::arith::FastMathFlagsAttr::get(
        rewriter.getContext(), mlir::arith::FastMathFlags::none
    );
    auto new_op = createFn(rewriter, op->getLoc(), input, fastmath);
    rewriter.replaceOp(op, new_op->getResults());
    return mlir::success();
}

mlir::LogicalResult
lowering_exp::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto exp = llvm::cast<exp_op>(op);
    return lowering_math_unary(op, exp.get_input(), rewriter,
        [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::arith::FastMathFlagsAttr fm) {
            return mlir::math::ExpOp::create(b, loc, v, fm);
        }
    );
}

mlir::LogicalResult
lowering_log::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto log = llvm::cast<log_op>(op);
    return lowering_math_unary(op, log.get_input(), rewriter,
        [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::arith::FastMathFlagsAttr fm) {
            return mlir::math::LogOp::create(b, loc, v, fm);
        }
    );
}

mlir::LogicalResult
lowering_sqrt::matchAndRewrite(mlir::Operation* op,
                               llvm::ArrayRef<mlir::Value> operands,
                               mlir::ConversionPatternRewriter& rewriter) const {
    auto sqrt = llvm::cast<sqrt_op>(op);
    return lowering_math_unary(op, sqrt.get_input(), rewriter,
        [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::arith::FastMathFlagsAttr fm) {
            return mlir::math::SqrtOp::create(b, loc, v, fm);
        }
    );
}

mlir::LogicalResult
lowering_tanh::matchAndRewrite(mlir::Operation* op,
                               llvm::ArrayRef<mlir::Value> operands,
                               mlir::ConversionPatternRewriter& rewriter) const {
    auto tanh = llvm::cast<tanh_op>(op);
    return lowering_math_unary(op, tanh.get_input(), rewriter,
        [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::arith::FastMathFlagsAttr fm) {
            return mlir::math::TanhOp::create(b, loc, v, fm);
        }
    );
}

mlir::LogicalResult
lowering_sigmoid::matchAndRewrite(mlir::Operation* op,
                                  llvm::ArrayRef<mlir::Value> operands,
                                  mlir::ConversionPatternRewriter& rewriter) const {
    auto sigmoid = llvm::cast<sigmoid_op>(op);
    auto input = sigmoid.get_input();
    auto ty = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto base_type = ty.getElementType();

    if (!llvm::isa<mlir::FloatType>(base_type)) {
        return mlir::failure();
    }

    auto ft = llvm::cast<mlir::FloatType>(base_type);
    mlir::TypedAttr zero_attr = mlir::DenseElementsAttr::get(
        ty, llvm::APFloat(ft.getFloatSemantics(), "0")
    );
    mlir::TypedAttr one_attr = mlir::DenseElementsAttr::get(
        ty, llvm::APFloat(ft.getFloatSemantics(), "1")
    );

    auto zero = mlir::arith::ConstantOp::create(rewriter, op->getLoc(), zero_attr);
    auto one = mlir::arith::ConstantOp::create(rewriter, op->getLoc(), one_attr);

    // -x
    auto neg_x = mlir::arith::SubFOp::create(
        rewriter, op->getLoc(), mlir::ValueRange { zero, input }
    );

    // exp(-x)
    auto fastmath = mlir::arith::FastMathFlagsAttr::get(
        rewriter.getContext(), mlir::arith::FastMathFlags::none
    );
    auto exp_neg = mlir::math::ExpOp::create(
        rewriter, op->getLoc(), neg_x->getResult(0), fastmath
    );

    // 1.0 + exp(-x)
    auto one_plus_exp = mlir::arith::AddFOp::create(
        rewriter, op->getLoc(),
        mlir::ValueRange { one, exp_neg->getResult(0) }
    );

    // 1.0 / (1.0 + exp(-x))
    auto result = mlir::arith::DivFOp::create(
        rewriter, op->getLoc(),
        mlir::ValueRange { one, one_plus_exp->getResult(0) }
    );

    rewriter.replaceOp(op, result->getResults());
    return mlir::success();
}

mlir::LogicalResult
lowering_yield::matchAndRewrite(mlir::Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                mlir::ConversionPatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, operands);
    return mlir::success();
}

mlir::LogicalResult
lowering_if::matchAndRewrite(mlir::Operation* op,
                             llvm::ArrayRef<mlir::Value> operands,
                             mlir::ConversionPatternRewriter& rewriter) const {
    auto ifop = llvm::cast<if_op>(op);
    auto loc = op->getLoc();

    auto scf_if = mlir::scf::IfOp::create(
        rewriter, loc, ifop.getResultTypes(), operands[0],
        /*withElseRegion=*/true
    );

    // then region: inline colgm.if's then block into scf.if's then region
    rewriter.inlineRegionBefore(ifop.get_then_region(),
                                scf_if.getThenRegion(),
                                scf_if.getThenRegion().begin());
    scf_if.getThenRegion().back().erase(); // remove default empty block

    // else region: inline colgm.if's else block into scf.if's else region
    rewriter.inlineRegionBefore(ifop.get_else_region(),
                                scf_if.getElseRegion(),
                                scf_if.getElseRegion().begin());
    scf_if.getElseRegion().back().erase(); // remove default empty block

    rewriter.replaceOp(op, scf_if.getResults());
    return mlir::success();
}

mlir::LogicalResult
lowering_for::matchAndRewrite(mlir::Operation* op,
                             llvm::ArrayRef<mlir::Value> operands,
                             mlir::ConversionPatternRewriter& rewriter) const {
    auto forop = llvm::cast<for_op>(op);
    auto loc = op->getLoc();
    auto upper_bound = forop.get_upper_bound();
    auto lower_bound = forop.get_lower_bound();
    auto init_args = forop.get_iter_args();
    auto step = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(1)
    );

    auto scf_for = mlir::scf::ForOp::create(
        rewriter, loc, lower_bound, upper_bound, step, init_args
    );

    // body region: inline colgm.for's body block into scf.for's body region
    rewriter.inlineRegionBefore(forop.get_body(),
                                scf_for.getBodyRegion(),
                                scf_for.getBodyRegion().begin());
    scf_for.getBodyRegion().back().erase(); // remove default empty block

    rewriter.replaceOp(op, scf_for.getResults());
    return mlir::success();
}

mlir::LogicalResult
lowering_matmul::matchAndRewrite(mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> operands,
                                 mlir::ConversionPatternRewriter& rewriter) const {
    auto matmul = llvm::cast<matmul_op>(op);
    auto lhs = matmul.get_lhs();
    auto rhs = matmul.get_rhs();
    auto loc = matmul.getLoc();
    auto result_type = llvm::cast<mlir::RankedTensorType>(matmul->getResult(0).getType());

    auto rank = result_type.getRank();
    auto elem_type = result_type.getElementType();

    // create empty output tensor
    auto init = mlir::tensor::EmptyOp::create(
        rewriter, loc, result_type.getShape(), elem_type
    );

    if (rank == 2) {
        auto matmul_2d = mlir::linalg::MatmulOp::create(
            rewriter, loc,
            mlir::ValueRange { lhs, rhs },
            mlir::ValueRange { init }
        );
        rewriter.replaceOp(op, matmul_2d->getResults());
    } else {
        auto batch_matmul = mlir::linalg::BatchMatmulOp::create(
            rewriter, loc,
            mlir::ValueRange { lhs, rhs },
            mlir::ValueRange { init }
        );
        rewriter.replaceOp(op, batch_matmul->getResults());
    }

    return mlir::success();
}

mlir::LogicalResult
lowering_elements::matchAndRewrite(mlir::Operation* op,
                                   llvm::ArrayRef<mlir::Value> operands,
                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto elem = llvm::cast<elements_op>(op);
    auto loc = op->getLoc();
    auto num_ops = elem->getNumOperands();

    // extract scalars from each rank-0 operand
    llvm::SmallVector<mlir::Value> scalars;
    for (unsigned i = 0; i < num_ops; ++i) {
        auto extract = mlir::tensor::ExtractOp::create(
            rewriter, loc, elem->getOperand(i), mlir::ValueRange{}
        );
        scalars.push_back(extract);
    }

    auto result_type = llvm::cast<mlir::RankedTensorType>(elem->getResult(0).getType());
    auto elt_type = result_type.getElementType();
    auto flat_type = mlir::RankedTensorType::get({static_cast<i64>(num_ops)}, elt_type);

    auto fe = mlir::tensor::FromElementsOp::create(rewriter, loc, flat_type, scalars);

    if (result_type.getRank() == 1) {
        rewriter.replaceOp(op, fe->getResults());
    } else {
        auto rank = result_type.getRank();
        auto shape_type = mlir::RankedTensorType::get({rank}, rewriter.getIndexType());
        auto shape_attr = mlir::DenseIntElementsAttr::get(shape_type, result_type.getShape());
        auto shape_val = mlir::arith::ConstantOp::create(rewriter, loc, shape_attr);
        auto reshape = mlir::tensor::ReshapeOp::create(
            rewriter, loc, result_type, fe->getResult(0), shape_val
        );
        rewriter.replaceOp(op, reshape->getResults());
    }

    return mlir::success();
}

mlir::LogicalResult
lowering_stack::matchAndRewrite(mlir::Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                mlir::ConversionPatternRewriter& rewriter) const {
    auto stack = llvm::cast<stack_op>(op);
    auto loc = op->getLoc();
    auto target_attr = stack.get_target_shape();

    // decode target_shape
    llvm::SmallVector<i64> target_shape;
    for (auto a : target_attr) {
        target_shape.push_back(llvm::cast<mlir::IntegerAttr>(a).getInt());
    }

    auto result_type = llvm::cast<mlir::RankedTensorType>(stack->getResult(0).getType());
    auto elt_type = result_type.getElementType();
    auto result_rank = result_type.getRank();
    i64 target_rank = target_shape.size();
    i64 operand_rank = result_rank - target_rank;

    // operand shape
    auto operand_type = llvm::cast<mlir::RankedTensorType>(stack->getOperand(0).getType());
    auto operand_shape = operand_type.getShape();

    // create empty result tensor
    auto empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, result_type.getShape(), elt_type
    );

    mlir::Value curr = empty;

    // for each operand, compute multi-dim position in target_shape,
    // expand rank to result_rank, then insert_slice
    i64 num_operands = stack->getNumOperands();
    for (i64 n = 0; n < num_operands; ++n) {
        // linear index n → multi-dimensional position
        llvm::SmallVector<i64> pos(target_rank);
        i64 remaining = n;
        for (i64 d = target_rank - 1; d >= 0; --d) {
            pos[d] = remaining % target_shape[d];
            remaining /= target_shape[d];
        }

        // reshape operand to have result_rank (add leading 1s)
        llvm::SmallVector<i64> expanded_shape;
        for (i64 d = 0; d < target_rank; ++d) expanded_shape.push_back(1);
        for (auto s : operand_shape) expanded_shape.push_back(s);

        auto expanded_type = mlir::RankedTensorType::get(expanded_shape, elt_type);
        auto shape_type = mlir::RankedTensorType::get({result_rank}, rewriter.getIndexType());
        auto shape_attr = mlir::DenseIntElementsAttr::get(shape_type, expanded_shape);
        auto shape_val = mlir::arith::ConstantOp::create(rewriter, loc, shape_attr);
        auto expanded = mlir::tensor::ReshapeOp::create(
            rewriter, loc, expanded_type, stack->getOperand(n), shape_val
        );

        // insert_slice
        llvm::SmallVector<mlir::OpFoldResult> offsets(result_rank, rewriter.getIndexAttr(0));
        llvm::SmallVector<mlir::OpFoldResult> sizes(result_rank);
        llvm::SmallVector<mlir::OpFoldResult> strides(result_rank, rewriter.getIndexAttr(1));

        for (i64 d = 0; d < target_rank; ++d) {
            offsets[d] = rewriter.getIndexAttr(pos[d]);
            sizes[d] = rewriter.getIndexAttr(1);
        }
        for (i64 d = 0; d < operand_rank; ++d) {
            sizes[target_rank + d] = rewriter.getIndexAttr(operand_shape[d]);
        }

        curr = mlir::tensor::InsertSliceOp::create(
            rewriter, loc, expanded->getResult(0), curr, offsets, sizes, strides
        );
    }

    rewriter.replaceOp(op, curr);
    return mlir::success();
}

mlir::LogicalResult
lowering_reshape::matchAndRewrite(mlir::Operation* op,
                                  llvm::ArrayRef<mlir::Value> operands,
                                  mlir::ConversionPatternRewriter& rewriter) const {
    auto reshape = llvm::cast<reshape_op>(op);
    auto loc = reshape.getLoc();
    auto input = reshape.get_input();
    auto result_type = llvm::cast<mlir::RankedTensorType>(reshape->getResult(0).getType());
    auto shape = reshape.get_target_shape();

    // decode target_shape
    llvm::SmallVector<i64> target_shape;
    for (auto a : shape) {
        target_shape.push_back(llvm::cast<mlir::IntegerAttr>(a).getInt());
    }

    auto rank = static_cast<long long>(shape.size());
    auto shape_type = mlir::RankedTensorType::get({rank}, rewriter.getIndexType());
    auto shape_attr = mlir::DenseIntElementsAttr::get(shape_type, target_shape);
    auto shape_val = mlir::arith::ConstantOp::create(rewriter, loc, shape_attr);

    auto rs = mlir::tensor::ReshapeOp::create(rewriter, loc, result_type, input, shape_val);
    rewriter.replaceOp(op, rs->getResults());
    return mlir::success();
}

mlir::LogicalResult
lowering_transpose::matchAndRewrite(mlir::Operation* op,
                                    llvm::ArrayRef<mlir::Value> operands,
                                    mlir::ConversionPatternRewriter& rewriter) const {
    auto transpose = llvm::cast<transpose_op>(op);
    auto loc = transpose.getLoc();
    auto input = transpose.get_input();
    auto permutation = transpose.get_permutation();
    auto result_type = llvm::cast<mlir::RankedTensorType>(transpose->getResult(0).getType());

    llvm::SmallVector<i64> perm;
    for (auto a : permutation) {
        perm.push_back(llvm::cast<mlir::IntegerAttr>(a).getInt());
    }

    auto out_init = mlir::tensor::EmptyOp::create(
        rewriter, loc, result_type.getShape(), result_type.getElementType()
    );
    auto trans = mlir::linalg::TransposeOp::create(
        rewriter, loc, input, out_init, perm
    );
    rewriter.replaceOp(op, trans->getResults());
    return mlir::success();
}

mlir::LogicalResult
lowering_broadcast::matchAndRewrite(mlir::Operation* op,
                                    llvm::ArrayRef<mlir::Value> operands,
                                    mlir::ConversionPatternRewriter& rewriter) const {
    auto broadcast = llvm::cast<broadcast_op>(op);
    auto loc = broadcast.getLoc();
    auto input = broadcast.get_input();

    auto result_type = llvm::cast<mlir::RankedTensorType>(broadcast->getResult(0).getType());
    auto elem_type = result_type.getElementType();
    auto output_shape = result_type.getShape();
    auto output_rank = static_cast<long long>(output_shape.size());

    auto target_attr = broadcast.get_target_shape();
    llvm::SmallVector<i64> target;
    for (auto a : target_attr) {
        target.push_back(llvm::cast<mlir::IntegerAttr>(a).getInt());
    }

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto input_shape = input_type.getShape();
    auto input_rank = static_cast<long long>(input_shape.size());

    bool is_identity = (input_rank == output_rank);
    for (i64 i = 0; i < output_rank && is_identity; ++i) {
        is_identity &= (input_shape[i] == target[i]);
    }
    // all dims match, directly pass through input
    if (is_identity) {
        rewriter.replaceOp(op, input);
        return mlir::success();
    }

    auto ctx = rewriter.getContext();
    auto src_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto src_shape = src_type.getShape();

    // build input affine map
    llvm::SmallVector<mlir::AffineExpr> input_exprs;
    if (input_rank != 0) {
        auto offset = output_rank - input_rank;
        for (i64 k = 0; k < input_rank; ++k) {
            auto i = offset + k;
            if (src_shape[k] == 1 && target[i] > 1) {
                // existing dim, size 1 => N
                input_exprs.push_back(mlir::getAffineConstantExpr(0, ctx));
            } else {
                // dims match
                input_exprs.push_back(mlir::getAffineDimExpr(i, ctx));
            }
        }
    }
    auto input_map = mlir::AffineMap::get(output_rank, 0, input_exprs, ctx);

    // output affine map
    llvm::SmallVector<mlir::AffineExpr> output_exprs;
    for (i64 i = 0; i < output_rank; ++i) {
        output_exprs.push_back(mlir::getAffineDimExpr(i, ctx));
    }
    auto output_map = mlir::AffineMap::get(output_rank, 0, output_exprs, ctx);

    llvm::SmallVector<mlir::utils::IteratorType> iter_types(
        output_rank, mlir::utils::IteratorType::parallel
    );

    auto empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, output_shape, elem_type
    );
    auto generic = mlir::linalg::GenericOp::create(
        rewriter, loc,
        mlir::TypeRange { result_type },
        mlir::ValueRange { input }, mlir::ValueRange { empty },
        llvm::ArrayRef<mlir::AffineMap> { input_map, output_map },
        iter_types,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
            mlir::linalg::YieldOp::create(b, loc, args[0]);
        }
    );
    rewriter.replaceOp(op, generic->getResults());
    return mlir::success();
}

mlir::LogicalResult
lowering_reduce_sum::matchAndRewrite(mlir::Operation* op,
                                     llvm::ArrayRef<mlir::Value> operands,
                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto rs = llvm::cast<reduce_sum>(op);
    auto loc = rs.getLoc();
    auto input = rs.get_input();
    auto axes_attr = rs.get_axes();
    auto result_type = llvm::cast<mlir::RankedTensorType>(rs->getResult(0).getType());

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto input_shape = input_type.getShape();
    auto input_rank = static_cast<i64>(input_shape.size());
    auto elem_type = input_type.getElementType();

    llvm::SmallBitVector reduced(static_cast<unsigned>(input_rank));
    for (auto a : axes_attr) {
        reduced.set(static_cast<unsigned>(llvm::cast<mlir::IntegerAttr>(a).getInt()));
    }

    llvm::SmallVector<mlir::utils::IteratorType> iter_types;
    for (i64 i = 0; i < input_rank; ++i) {
        if (reduced[i]) {
            iter_types.push_back(mlir::utils::IteratorType::reduction);
        } else {
            iter_types.push_back(mlir::utils::IteratorType::parallel);
        }
    }

    auto ctx = rewriter.getContext();

    llvm::SmallVector<mlir::AffineExpr> input_exprs;
    for (i64 i = 0; i < input_rank; ++i) {
        input_exprs.push_back(mlir::getAffineDimExpr(i, ctx));
    }
    auto input_map = mlir::AffineMap::get(input_rank, 0, input_exprs, ctx);

    llvm::SmallVector<mlir::AffineExpr> output_exprs;
    for (i64 i = 0; i < input_rank; ++i) {
        if (!reduced[i]) {
            output_exprs.push_back(mlir::getAffineDimExpr(i, ctx));
        }
    }
    auto output_map = mlir::AffineMap::get(input_rank, 0, output_exprs, ctx);

    auto empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, result_type.getShape(), elem_type
    );

    auto generic = mlir::linalg::GenericOp::create(
        rewriter, loc,
        mlir::TypeRange { result_type },
        mlir::ValueRange { input }, mlir::ValueRange { empty },
        llvm::ArrayRef<mlir::AffineMap> { input_map, output_map },
        iter_types,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value acc;
            if (llvm::isa<mlir::FloatType>(elem_type)) {
                auto add = mlir::arith::AddFOp::create(b, loc, mlir::ValueRange{args[0], args[1]});
                acc = add->getResult(0);
            } else if (llvm::isa<mlir::IntegerType>(elem_type)) {
                auto add = mlir::arith::AddIOp::create(b, loc, mlir::ValueRange{args[0], args[1]});
                acc = add->getResult(0);
            } else {
                return;
            }
            mlir::linalg::YieldOp::create(b, loc, acc);
        }
    );
    rewriter.replaceOp(op, generic->getResults());
    return mlir::success();
}

static void emit_print_scalar(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Value scalar, mlir::Type elem_type) {
    llvm::StringRef callee;
    if (llvm::isa<mlir::FloatType>(elem_type)) {
        callee = elem_type.isF64() ? "__colgm_print_f64" : "__colgm_print_f32";
    } else if (auto it = llvm::dyn_cast<mlir::IntegerType>(elem_type)) {
        if (it.getWidth() == 1)       callee = "__colgm_print_i1";
        else if (it.getWidth() == 64) callee = "__colgm_print_i64";
        else                          callee = "__colgm_print_i32";
    } else {
        return;
    }
    mlir::func::CallOp::create(builder, loc, /*results=*/{}, callee,
                               mlir::ValueRange{scalar});
}

static void emit_print_tensor(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Value tensor,
                              llvm::ArrayRef<i64> shape, mlir::Type elem_type,
                              llvm::SmallVectorImpl<mlir::Value>& indices,
                              unsigned dim) {
    i64 rank = static_cast<i64>(shape.size());

    if (dim == static_cast<unsigned>(rank)) {
        auto extract = mlir::tensor::ExtractOp::create(builder, loc, tensor, indices);
        emit_print_scalar(builder, loc, extract, elem_type);
        return;
    }

    mlir::func::CallOp::create(builder, loc, {}, "__colgm_print_open_bracket", {});

    i64 size = shape[dim];
    auto c0 = mlir::arith::ConstantOp::create(builder, loc, builder.getIndexAttr(0));
    auto c1 = mlir::arith::ConstantOp::create(builder, loc, builder.getIndexAttr(1));
    auto ub = mlir::arith::ConstantOp::create(builder, loc, builder.getIndexAttr(size));

    auto for_op = mlir::scf::ForOp::create(builder, loc, c0, ub, c1);

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(for_op.getBody());
        auto iv = for_op.getInductionVar();

        auto is_not_zero = mlir::arith::CmpIOp::create(
            builder, loc,
            mlir::arith::CmpIPredicate::ne, iv, c0);
        mlir::scf::IfOp::create(builder, loc, is_not_zero,
            [&](mlir::OpBuilder& then_b, mlir::Location then_loc) {
                mlir::func::CallOp::create(then_b, then_loc, {},
                                           "__colgm_print_comma", {});
                mlir::scf::YieldOp::create(then_b, then_loc);
            });

        indices.push_back(iv);
        emit_print_tensor(builder, loc, tensor, shape, elem_type,
                          indices, dim + 1);
        indices.pop_back();
    }

    mlir::func::CallOp::create(builder, loc, {}, "__colgm_print_close_bracket", {});
}

mlir::LogicalResult
lowering_print::matchAndRewrite(mlir::Operation* op,
                                llvm::ArrayRef<mlir::Value> operands,
                                mlir::ConversionPatternRewriter& rewriter) const {
    auto print = llvm::cast<print_op>(op);
    auto loc = print.getLoc();

    // declare external print runtime functions in the parent module
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (module) {
        auto ctx = rewriter.getContext();
        auto void_ty = mlir::FunctionType::get(ctx, {}, {});
        auto f32_fn_ty = mlir::FunctionType::get(ctx, {mlir::Float32Type::get(ctx)}, {});
        auto f64_fn_ty = mlir::FunctionType::get(ctx, {mlir::Float64Type::get(ctx)}, {});
        auto i1_fn_ty = mlir::FunctionType::get(ctx, {mlir::IntegerType::get(ctx, 1)}, {});
        auto i32_fn_ty = mlir::FunctionType::get(ctx, {mlir::IntegerType::get(ctx, 32)}, {});
        auto i64_fn_ty = mlir::FunctionType::get(ctx, {mlir::IntegerType::get(ctx, 64)}, {});

        auto declare_extern = [&](llvm::StringRef name, mlir::FunctionType fn_ty) {
            if (module.lookupSymbol<mlir::func::FuncOp>(name)) {
                return;
            }
            mlir::OpBuilder b(ctx);
            b.setInsertionPointToStart(module.getBody());
            auto fn = mlir::func::FuncOp::create(b, b.getUnknownLoc(), name, fn_ty);
            fn.setPrivate();
        };
        declare_extern("__colgm_print_open_bracket", void_ty);
        declare_extern("__colgm_print_close_bracket", void_ty);
        declare_extern("__colgm_print_comma", void_ty);
        declare_extern("__colgm_print_newline", void_ty);
        declare_extern("__colgm_print_f32", f32_fn_ty);
        declare_extern("__colgm_print_f64", f64_fn_ty);
        declare_extern("__colgm_print_i1", i1_fn_ty);
        declare_extern("__colgm_print_i32", i32_fn_ty);
        declare_extern("__colgm_print_i64", i64_fn_ty);
    }

    rewriter.setInsertionPoint(op);

    i64 num_operands = print->getNumOperands();
    for (i64 i = 0; i < num_operands; ++i) {
        auto tensor = print->getOperand(i);
        auto tensor_type = llvm::cast<mlir::RankedTensorType>(tensor.getType());
        auto shape = tensor_type.getShape();
        auto elem_type = tensor_type.getElementType();

        llvm::SmallVector<mlir::Value> indices;
        emit_print_tensor(rewriter, loc, tensor, shape, elem_type, indices, 0);

        if (i < num_operands - 1) {
            mlir::func::CallOp::create(rewriter, loc, {},
                                       "__colgm_print_newline", {});
        }
    }

    rewriter.eraseOp(op);
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
                 lowering_cast, lowering_slice,
                 lowering_cmp_eq, lowering_cmp_ne,
                 lowering_cmp_lt, lowering_cmp_le,
                 lowering_cmp_gt, lowering_cmp_ge,
                 lowering_neg, lowering_relu,
                 lowering_abs, lowering_exp,
                 lowering_log, lowering_sqrt,
                 lowering_tanh, lowering_sigmoid,
                 lowering_yield, lowering_if, lowering_for,
                 lowering_matmul,
                 lowering_elements, lowering_stack,
                 lowering_reshape, lowering_transpose,
                 lowering_broadcast, lowering_reduce_sum,
                 lowering_print>(cvt, &getContext());

    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, frozen))) {
        signalPassFailure();
    }
}

static mlir::PassRegistration<colgm_lowering> pass;

}
