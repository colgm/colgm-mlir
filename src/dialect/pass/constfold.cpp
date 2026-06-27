#include "dialect/colgm/elements.hpp"
#include "dialect/colgm/stack.hpp"
#include "dialect/colgm/constant_op.hpp"
#include "dialect/pass/constfold.hpp"

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> createColgmConstFoldPass() {
    return std::make_unique<colgm_const_fold_pass>();
}

// shared folding logic for ops that pack N constants into one constant
template <typename OpTy>
static void tryFoldPackOp(OpTy elem) {
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
        return;
    }

    llvm::SmallVector<mlir::Attribute> elt_attrs;
    for (auto& val : const_values) {
        for (auto elt : val.getValues<mlir::Attribute>()) {
            elt_attrs.push_back(elt);
        }
    }

    auto result_type = llvm::cast<mlir::RankedTensorType>(
        elem->getResult(0).getType());

    auto folded = mlir::DenseElementsAttr::get(result_type, elt_attrs);
    mlir::OpBuilder builder(elem);
    auto new_const = constant_op::create(builder, elem.getLoc(), folded, result_type);

    elem.replaceAllUsesWith(new_const);
    elem.erase();
    for (auto op : source_ops) {
        if (op->use_empty()) {
            op.erase();
        }
    }
}

void colgm_const_fold_pass::runOnOperation() {
    getOperation().walk([&](elements_op elem) {
        tryFoldPackOp(elem);
    });
    getOperation().walk([&](stack_op elem) {
        tryFoldPackOp(elem);
    });
}

static mlir::PassRegistration<colgm_const_fold_pass> pass;

}