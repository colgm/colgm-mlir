#include <mlir/IR/OpImplementation.h>

#include "dialect/op.hpp"

namespace colgm_mlir {

void add_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

mlir::ParseResult add_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void add_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

mlir::LogicalResult add_op::verify() {
    return mlir::success();
}
    

}