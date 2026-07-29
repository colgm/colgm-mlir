#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>

#include "dialect/pass/lowering/unary_op.hpp"

namespace colgm_mlir {

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
lowering_sin::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto sin = llvm::cast<sin_op>(op);
    return lowering_math_unary(op, sin.get_input(), rewriter,
        [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::arith::FastMathFlagsAttr fm) {
            return mlir::math::SinOp::create(b, loc, v, fm);
        }
    );
}

mlir::LogicalResult
lowering_cos::matchAndRewrite(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::ConversionPatternRewriter& rewriter) const {
    auto cos = llvm::cast<cos_op>(op);
    return lowering_math_unary(op, cos.get_input(), rewriter,
        [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::arith::FastMathFlagsAttr fm) {
            return mlir::math::CosOp::create(b, loc, v, fm);
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
lowering_gelu::matchAndRewrite(mlir::Operation* op,
                               llvm::ArrayRef<mlir::Value> operands,
                               mlir::ConversionPatternRewriter& rewriter) const {
    auto gelu = llvm::cast<gelu_op>(op);
    auto input = gelu.get_input();
    auto ty = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto base_type = ty.getElementType();

    if (!llvm::isa<mlir::FloatType>(base_type)) {
        return mlir::failure();
    }

    auto ft = llvm::cast<mlir::FloatType>(base_type);
    auto loc = op->getLoc();
    auto fastmath = mlir::arith::FastMathFlagsAttr::get(
        rewriter.getContext(), mlir::arith::FastMathFlags::none
    );

    // Constants
    mlir::TypedAttr one_attr = mlir::DenseElementsAttr::get(
        ty, llvm::APFloat(ft.getFloatSemantics(), "1")
    );
    mlir::TypedAttr half_attr = mlir::DenseElementsAttr::get(
        ty, llvm::APFloat(ft.getFloatSemantics(), "0.5")
    );
    mlir::TypedAttr coeff1_attr = mlir::DenseElementsAttr::get(
        ty, llvm::APFloat(ft.getFloatSemantics(), "0.044715")
    );
    // sqrt(2/pi) ≈ 0.7978845608028654
    mlir::TypedAttr coeff2_attr = mlir::DenseElementsAttr::get(
        ty, llvm::APFloat(ft.getFloatSemantics(), "0.7978845608028654")
    );

    auto one = mlir::arith::ConstantOp::create(rewriter, loc, one_attr);
    auto half = mlir::arith::ConstantOp::create(rewriter, loc, half_attr);
    auto coeff1 = mlir::arith::ConstantOp::create(rewriter, loc, coeff1_attr);
    auto coeff2 = mlir::arith::ConstantOp::create(rewriter, loc, coeff2_attr);

    // x^3 = x * x * x
    auto x2 = mlir::arith::MulFOp::create(
        rewriter, loc, mlir::ValueRange{ input, input }
    );
    auto x3 = mlir::arith::MulFOp::create(
        rewriter, loc, mlir::ValueRange{ x2->getResult(0), input }
    );

    // 0.044715 * x^3
    auto t1 = mlir::arith::MulFOp::create(
        rewriter, loc, mlir::ValueRange{ coeff1, x3->getResult(0) }
    );

    // x + 0.044715 * x^3
    auto t2 = mlir::arith::AddFOp::create(
        rewriter, loc, mlir::ValueRange{ input, t1->getResult(0) }
    );

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    auto t3 = mlir::arith::MulFOp::create(
        rewriter, loc, mlir::ValueRange{ coeff2, t2->getResult(0) }
    );

    // tanh(...)
    auto tanh = mlir::math::TanhOp::create(
        rewriter, loc, t3->getResult(0), fastmath
    );

    // 1 + tanh(...)
    auto t4 = mlir::arith::AddFOp::create(
        rewriter, loc, mlir::ValueRange{ one, tanh->getResult(0) }
    );

    // 0.5 * (1 + tanh(...))
    auto t5 = mlir::arith::MulFOp::create(
        rewriter, loc, mlir::ValueRange{ half, t4->getResult(0) }
    );

    // x * 0.5 * (1 + tanh(...))
    auto result = mlir::arith::MulFOp::create(
        rewriter, loc, mlir::ValueRange{ input, t5->getResult(0) }
    );

    rewriter.replaceOp(op, result->getResults());
    return mlir::success();
}

}
