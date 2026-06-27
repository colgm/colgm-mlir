#include "dialect/colgm/elements.hpp"
#include "dialect/colgm/constant_op.hpp"
#include "dialect/pass/constfold.hpp"

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> createColgmConstFoldPass() {
    return std::make_unique<colgm_const_fold_pass>();
}

void colgm_const_fold_pass::runOnOperation() {
    auto& ctx = getContext();
    getOperation().walk([&](elements_op elem) {
        llvm::SmallVector<mlir::DenseElementsAttr> const_values;
        llvm::SmallVector<constant_op> source_ops;
        bool all_const = true;
        for (auto operand : elem->getOperands()) {
            auto def_op = operand.getDefiningOp<constant_op>();
            if (!def_op) {
                all_const = false;
                break;
            }

            // constants generated from colgm-mlir DSL itself make sure they are
            // all dense elements, so we can use DenseElementsAttr
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

        // extract dense's element, compatible for all ranks
        llvm::SmallVector<mlir::Attribute> elt_attrs;
        for (auto& val : const_values) {
            for (auto elt : val.getValues<mlir::Attribute>()) {
                elt_attrs.push_back(elt);
            }
        }

        auto elt_type = llvm::cast<mlir::RankedTensorType>(
            elem->getResult(0).getType()).getElementType();
        auto shape = elem.get_target_shape();
        llvm::SmallVector<i64> dims;
        for (auto dim : shape) {
            dims.push_back(llvm::cast<mlir::IntegerAttr>(dim).getInt());
        }
        auto result_type = mlir::RankedTensorType::get(dims, elt_type);

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
    });
}

static mlir::PassRegistration<colgm_const_fold_pass> pass;

}