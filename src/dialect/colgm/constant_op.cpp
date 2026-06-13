#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/constant_op.hpp"

namespace colgm_mlir {

void constant_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        mlir::TypedAttr value, mlir::Type type) {
    state.addAttribute("value", value);
    state.addTypes(type);
}

void constant_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        i64 value, mlir::Type type) {
    auto tensor_type = llvm::cast<mlir::RankedTensorType>(type);
    auto attr = builder.getIntegerAttr(tensor_type.getElementType(), value);
    state.addAttribute("value", attr);
    state.addTypes(type);
}

void constant_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        f64 value, mlir::Type type) {
    auto tensor_type = llvm::cast<mlir::RankedTensorType>(type);
    auto attr = builder.getFloatAttr(tensor_type.getElementType(), value);
    state.addAttribute("value", attr);
    state.addTypes(type);
}

void constant_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        bool value, mlir::Type type) {
    auto attr = builder.getBoolAttr(value);
    state.addAttribute("value", attr);
    state.addTypes(type);
}

mlir::ParseResult constant_op::parse(mlir::OpAsmParser& parser,
                                     mlir::OperationState& result) {
    mlir::TypedAttr value;

    if (parser.parseAttribute(value, "value", result.attributes)) {
        return mlir::failure();
    }

    result.addTypes(value.getType());
    return mlir::success();
}

void constant_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_value();
}

mlir::LogicalResult constant_op::verify() {
    auto value = get_value();
    auto result_type = (*this)->getResult(0).getType();

    if (value.getType() != result_type) {
        return emitOpError("value type ") << value.getType()
               << " does not match result type " << result_type;
    }
    return mlir::success();
}

}
