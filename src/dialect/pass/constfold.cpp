
#include "dialect/pass/constfold.hpp"

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> create_colgm_const_fold_pass() {
    return std::make_unique<colgm_const_fold_pass>();
}

// shared folding logic for ops that pack N constants into one constant
template <typename OpTy>
static bool try_fold_pack_op(OpTy elem, mlir::PatternRewriter& rewriter) {
    llvm::SmallVector<mlir::DenseElementsAttr> const_values;
    llvm::SmallVector<constant_op> source_ops;
    bool all_const = true;
    for (mlir::Value operand : elem->getOperands()) {
        auto def_op = operand.getDefiningOp<constant_op>();
        if (!def_op) {
            all_const = false;
            break;
        }

        auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(def_op.get_value());
        if (!attr) {
            all_const = false;
            break;
        }

        const_values.push_back(attr);
        source_ops.push_back(def_op);
    }

    if (!all_const) {
        return false;
    }

    llvm::SmallVector<mlir::Attribute> elt_attrs;
    for (auto& val : const_values) {
        for (auto elt : val.getValues<mlir::Attribute>()) {
            elt_attrs.push_back(elt);
        }
    }

    auto res_type = llvm::cast<mlir::RankedTensorType>(elem->getResult(0).getType());
    auto folded = mlir::DenseElementsAttr::get(res_type, elt_attrs);
    auto new_const = constant_op::create(rewriter, elem.getLoc(), folded, res_type);

    rewriter.replaceOp(elem, new_const);
    for (auto op : source_ops) {
        if (op->use_empty()) {
            rewriter.eraseOp(op);
        }
    }
    return true;
}

static bool try_fold_neg(neg_op elem, mlir::PatternRewriter& rewriter) {
    auto input = elem.get_input();
    auto def_op = input.getDefiningOp<constant_op>();
    if (!def_op) {
        return false;
    }

    auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(def_op.get_value());
    if (!attr) {
        return false;
    }

    llvm::SmallVector<mlir::Attribute> elt_attrs;
    for (auto elt : attr.getValues<mlir::Attribute>()) {
        if (llvm::isa<mlir::FloatAttr>(elt)) {
            auto tmp = llvm::cast<mlir::FloatAttr>(elt);
            auto val = -tmp.getValue();
            elt_attrs.push_back(mlir::FloatAttr::get(tmp.getType(), val));
        } else if (llvm::isa<mlir::IntegerAttr>(elt)) {
            auto tmp = llvm::cast<mlir::IntegerAttr>(elt);
            auto val = -tmp.getValue().getSExtValue();
            elt_attrs.push_back(mlir::IntegerAttr::get(tmp.getType(), val));
        } else {
            return false;
        }
    }

    auto res_type = llvm::cast<mlir::RankedTensorType>(def_op.get_value().getType());
    auto folded = mlir::DenseElementsAttr::get(res_type, elt_attrs);
    auto new_const = constant_op::create(rewriter, elem.getLoc(), folded, res_type);

    rewriter.replaceOp(elem, new_const);
    if (def_op->use_empty()) {
        rewriter.eraseOp(def_op);
    }
    return true;
}

template <typename OpTy>
static bool try_canonicalize_rhs_zero(OpTy elem, mlir::PatternRewriter& rewriter) {
    mlir::Value rhs = elem.get_rhs();
    auto constant = rhs.getDefiningOp<constant_op>();
    if (!constant) {
        return false;
    }

    auto dea = llvm::dyn_cast<mlir::DenseElementsAttr>(constant.get_value());
    if (!dea) {
        return false;
    }

    if (!dea.isSplat()) {
        return false;
    }

    auto type = dea.getElementType();
    if (auto int_ty = llvm::dyn_cast<mlir::IntegerType>(type)) {
        if (mlir::IntegerAttr::get(type, 0) == dea.getSplatValue<mlir::Attribute>()) {
            rewriter.replaceOp(elem, elem.get_lhs());
            rewriter.eraseOp(constant);
            return true;
        }
        return false;
    }

    if (auto float_ty = llvm::dyn_cast<mlir::FloatType>(type)) {
        if (mlir::FloatAttr::get(type, 0.0) == dea.getSplatValue<mlir::Attribute>()) {
            rewriter.replaceOp(elem, elem.get_lhs());
            rewriter.eraseOp(constant);
            return true;
        }
        return false;
    }

    return false;
}

template <typename OpTy>
static bool try_canonicalize_rhs_one(OpTy elem, mlir::PatternRewriter& rewriter) {
    mlir::Value rhs = elem.get_rhs();
    auto constant = rhs.getDefiningOp<constant_op>();
    if (!constant) {
        return false;
    }

    auto dea = llvm::dyn_cast<mlir::DenseElementsAttr>(constant.get_value());
    if (!dea) {
        return false;
    }

    if (!dea.isSplat()) {
        return false;
    }

    auto type = dea.getElementType();
    if (auto int_ty = llvm::dyn_cast<mlir::IntegerType>(type)) {
        if (mlir::IntegerAttr::get(type, 1) == dea.getSplatValue<mlir::Attribute>()) {
            rewriter.replaceOp(elem, elem.get_lhs());
            rewriter.eraseOp(constant);
            return true;
        }
        return false;
    }

    if (auto float_ty = llvm::dyn_cast<mlir::FloatType>(type)) {
        if (mlir::FloatAttr::get(type, 1.0) == dea.getSplatValue<mlir::Attribute>()) {
            rewriter.replaceOp(elem, elem.get_lhs());
            rewriter.eraseOp(constant);
            return true;
        }
        return false;
    }

    return false;
}

mlir::LogicalResult fold_neg::matchAndRewrite(neg_op elem,
                                              mlir::PatternRewriter& rewriter) const {
    return mlir::success(try_fold_neg(elem, rewriter));
}

mlir::LogicalResult fold_elements::matchAndRewrite(elements_op elem,
                                                   mlir::PatternRewriter& rewriter) const {
    return mlir::success(try_fold_pack_op(elem, rewriter));
}

mlir::LogicalResult fold_stack::matchAndRewrite(stack_op elem,
                                                mlir::PatternRewriter& rewriter) const {
    return mlir::success(try_fold_pack_op(elem, rewriter));
}

mlir::LogicalResult canonicalize_add::matchAndRewrite(add_op elem,
                                                      mlir::PatternRewriter& rewriter) const {
    // add x, 0 -> x
    // TODO: add 0, x -> x
    return mlir::success(try_canonicalize_rhs_zero(elem, rewriter));
}

mlir::LogicalResult canonicalize_sub::matchAndRewrite(sub_op elem,
                                                      mlir::PatternRewriter& rewriter) const {
    // sub x, 0 -> x
    // TODO: sub x, x -> 0
    return mlir::success(try_canonicalize_rhs_zero(elem, rewriter));
}

mlir::LogicalResult canonicalize_mul::matchAndRewrite(mul_op elem,
                                                      mlir::PatternRewriter& rewriter) const {
    // mul x, 1 -> x
    // TODO: mul 1, x -> x
    // TODO: mul x, 0 -> 0
    // TODO: mul 0, x -> 0
    return mlir::success(try_canonicalize_rhs_one(elem, rewriter));
}

mlir::LogicalResult canonicalize_div::matchAndRewrite(div_op elem,
                                                      mlir::PatternRewriter& rewriter) const {
    // div x, 1 -> x
    return mlir::success(try_canonicalize_rhs_one(elem, rewriter));
}

void colgm_const_fold_pass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<fold_neg, fold_elements, fold_stack,
                 canonicalize_add, canonicalize_sub,
                 canonicalize_mul, canonicalize_div>(&getContext());

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(32);

    (void)mlir::applyPatternsGreedily(getOperation(),
                                      std::move(patterns), config);
}

static mlir::PassRegistration<colgm_const_fold_pass> pass;

}