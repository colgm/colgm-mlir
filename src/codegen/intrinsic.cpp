#include "codegen/intrinsic.hpp"

namespace colgm_mlir {

intrinsic_generator_registry::intrinsic_generator_registry() {
    regist("relu", relu_gen);
    regist("abs", abs_gen);
    regist("exp", exp_gen);
    regist("log", log_gen);
    regist("sqrt", sqrt_gen);
    regist("tanh", tanh_gen);
    regist("sigmoid", sigmoid_gen);
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

}