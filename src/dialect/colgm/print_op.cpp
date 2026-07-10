#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/print_op.hpp"

namespace colgm_mlir {

void print_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                     mlir::ValueRange inputs) {
    state.addOperands(inputs);
}

print_op print_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::ValueRange inputs) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, inputs);
    return llvm::cast<print_op>(builder.create(state));
}

mlir::ParseResult print_op::parse(mlir::OpAsmParser& parser,
                                  mlir::OperationState& result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::Type> types;

    // parse: %a, %b : type_a, type_b   (comma-separated)
    // or: empty (no operands)
    if (parser.parseOperandList(operands)) {
        return mlir::failure();
    }
    if (!operands.empty()) {
        if (parser.parseColonTypeList(types)) {
            return mlir::failure();
        }
        if (operands.size() != types.size()) {
            return parser.emitError(parser.getCurrentLocation(),
                                    "operand count does not match type count");
        }
        if (parser.resolveOperands(operands, types, parser.getCurrentLocation(),
                                   result.operands)) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

void print_op::print(mlir::OpAsmPrinter& p) {
    if (!getOperands().empty()) {
        p << " ";
        p.printOperands(getOperands());
        p << " : ";
        llvm::interleaveComma(getOperands().getTypes(), p);
    }
}

}
