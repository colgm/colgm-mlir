#include "codegen/intrinsic.hpp"
#include <mlir/IR/BuiltinAttributes.h>

#include <functional>

namespace colgm_mlir {

// Recursively erase a Value's defining op and its dead operand chain.
// Handles both `colgm.constant` (direct) and `colgm.elements` (with
// rank-0 constant operands) — both paths from extract_i64_constants.
//
// Erase order matters: ancestors first, then children, to avoid
// dangling operand references (child results are operands of parent).
static void erase_dead_value(mlir::Value val) {
    auto* def_op = val.getDefiningOp();
    if (!def_op) return;

    // Post-order collect: leaves pushed first, ancestors last
    llvm::SmallVector<mlir::Operation*> chain;
    std::function<void(mlir::Operation*)> collect = [&](mlir::Operation* op) {
        for (auto operand : op->getOperands()) {
            if (auto* operand_op = operand.getDefiningOp()) {
                if (operand_op->hasOneUse()) {
                    collect(operand_op);
                }
            }
        }
        chain.push_back(op);
    };
    collect(def_op);

    // Erase ancestors first (reverse of collection order)
    for (auto* op : llvm::reverse(chain)) {
        op->erase();
    }
}

intrinsic_generator_registry::intrinsic_generator_registry() {
    regist("relu", relu_gen);
    regist("abs", abs_gen);
    regist("exp", exp_gen);
    regist("log", log_gen);
    regist("sqrt", sqrt_gen);
    regist("tanh", tanh_gen);
    regist("sigmoid", sigmoid_gen);
    regist("print", print_gen);
    regist("matmul", matmul_gen);
    regist("broadcast", broadcast_gen);
    regist("reduce_sum", reduce_sum_gen);
    regist("reshape", reshape_gen);
    regist("transpose", transpose_gen);
}

intrinsic_gen_find_res
intrinsic_generator_registry::find(const std::string& name) const {
    if (intrinsic_generators.find(name) != intrinsic_generators.end()) {
        return intrinsic_gen_find_res{ intrinsic_generators.at(name), true };
    }
    return intrinsic_gen_find_res { nullptr, false };
}

mlir::Value relu_gen(mlir::OpBuilder& builder,
                     mlir::Location loc,
                     llvm::SmallVector<mlir::Value>& args) {
    auto relu = relu_op::create(builder, loc, args[0]);
    return relu->getResult(0);
}

mlir::Value abs_gen(mlir::OpBuilder& builder,
                    mlir::Location loc,
                    llvm::SmallVector<mlir::Value>& args) {
    auto abs = abs_op::create(builder, loc, args[0]);
    return abs->getResult(0);
}

mlir::Value exp_gen(mlir::OpBuilder& builder,
                    mlir::Location loc,
                    llvm::SmallVector<mlir::Value>& args) {
    auto exp = exp_op::create(builder, loc, args[0]);
    return exp->getResult(0);
}

mlir::Value log_gen(mlir::OpBuilder& builder,
                    mlir::Location loc,
                    llvm::SmallVector<mlir::Value>& args) {
    auto log = log_op::create(builder, loc, args[0]);
    return log->getResult(0);
}

mlir::Value sqrt_gen(mlir::OpBuilder& builder,
                     mlir::Location loc,
                     llvm::SmallVector<mlir::Value>& args) {
    auto sqrt = sqrt_op::create(builder, loc, args[0]);
    return sqrt->getResult(0);
}

mlir::Value tanh_gen(mlir::OpBuilder& builder,
                     mlir::Location loc,
                     llvm::SmallVector<mlir::Value>& args) {
    auto tanh = tanh_op::create(builder, loc, args[0]);
    return tanh->getResult(0);
}

mlir::Value sigmoid_gen(mlir::OpBuilder& builder,
                        mlir::Location loc,
                        llvm::SmallVector<mlir::Value>& args) {
    auto sigmoid = sigmoid_op::create(builder, loc, args[0]);
    return sigmoid->getResult(0);
}

mlir::Value print_gen(mlir::OpBuilder& builder,
                      mlir::Location loc,
                      llvm::SmallVector<mlir::Value>& args) {
    auto print = print_op::create(builder, loc, args);
    return print->getResult(0);
}

mlir::Value matmul_gen(mlir::OpBuilder& builder,
                       mlir::Location loc,
                       llvm::SmallVector<mlir::Value>& args) {
    auto matmul = matmul_op::create(builder, loc, args[0], args[1]);
    return matmul->getResult(0);
}

static bool extract_i64_constants(mlir::Value val,
                                   llvm::SmallVectorImpl<i64>& out) {
    auto* def_op = val.getDefiningOp();
    if (!def_op) return false;

    if (auto const_op = llvm::dyn_cast<constant_op>(def_op)) {
        auto attr = const_op.get_value();
        if (auto dense = llvm::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
            for (auto v : dense.getValues<i64>()) {
                out.push_back(v);
            }
            return true;
        }
        return false;
    }

    if (auto elem_op = llvm::dyn_cast<elements_op>(def_op)) {
        for (auto operand : elem_op->getOperands()) {
            auto* op_def = operand.getDefiningOp();
            auto cst = llvm::dyn_cast_or_null<constant_op>(op_def);
            if (!cst) return false;
            auto attr = cst.get_value();
            auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(attr);
            if (!dense) return false;
            auto vals = dense.getValues<i64>();
            if (vals.empty()) return false;
            out.push_back(*vals.begin());
        }
        return true;
    }

    return false;
}

mlir::Value broadcast_gen(mlir::OpBuilder& builder,
                           mlir::Location loc,
                           llvm::SmallVector<mlir::Value>& args) {
    llvm::SmallVector<i64> shape;
    extract_i64_constants(args[1], shape);
    auto op = broadcast_op::create(builder, loc, args[0], shape);
    erase_dead_value(args[1]);
    return op->getResult(0);
}

mlir::Value reduce_sum_gen(mlir::OpBuilder& builder,
                            mlir::Location loc,
                            llvm::SmallVector<mlir::Value>& args) {
    llvm::SmallVector<i64> axes;
    extract_i64_constants(args[1], axes);
    auto op = reduce_sum::create(builder, loc, args[0], axes);
    erase_dead_value(args[1]);
    return op->getResult(0);
}

mlir::Value reshape_gen(mlir::OpBuilder& builder,
                         mlir::Location loc,
                         llvm::SmallVector<mlir::Value>& args) {
    llvm::SmallVector<i64> shape;
    extract_i64_constants(args[1], shape);
    auto op = reshape_op::create(builder, loc, args[0], shape);
    erase_dead_value(args[1]);
    return op->getResult(0);
}

mlir::Value transpose_gen(mlir::OpBuilder& builder,
                           mlir::Location loc,
                           llvm::SmallVector<mlir::Value>& args) {
    llvm::SmallVector<i64> perm;
    extract_i64_constants(args[1], perm);
    auto op = transpose_op::create(builder, loc, args[0], perm);
    erase_dead_value(args[1]);
    return op->getResult(0);
}

}