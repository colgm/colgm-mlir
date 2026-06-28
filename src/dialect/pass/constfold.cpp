
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

void colgm_const_fold_pass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<fold_neg, fold_elements, fold_stack>(&getContext());

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(32);

    (void)mlir::applyPatternsGreedily(getOperation(),
                                      std::move(patterns), config);
}

static mlir::PassRegistration<colgm_const_fold_pass> pass;

}