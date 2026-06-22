#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/yield.hpp"
#include "dialect/colgm/if_op.hpp"
#include "dialect/colgm/for_op.hpp"

namespace colgm_mlir {

void yield_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                     mlir::Value value) {
    if (value) {
        state.addOperands(value);
    }
}

yield_op yield_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::Value value) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, value);
    return llvm::cast<yield_op>(builder.create(state));
}

mlir::ParseResult yield_op::parse(mlir::OpAsmParser& parser,
                                  mlir::OperationState& result) {
    // colgm.yield [%value : type]
    mlir::OpAsmParser::UnresolvedOperand operand;
    if (!parser.parseOptionalOperand(operand).has_value()) {
        return mlir::success();
    }

    mlir::Type type;
    if (parser.parseColonType(type)) {
        return mlir::failure();
    }
    if (parser.resolveOperand(operand, type, result.operands)) {
        return mlir::failure();
    }
    return mlir::success();
}

void yield_op::print(mlir::OpAsmPrinter& p) {
    if (getNumOperands() > 0) {
        p << " " << getOperand(0) << " : " << getOperand(0).getType();
    }
}

mlir::LogicalResult yield_op::verify() {
    if (getNumOperands() > 1) {
        return emitOpError("yield can have at most one operand");
    }

    auto parent = (*this)->getParentOp();
    if (!llvm::isa<if_op>(parent) && !llvm::isa<for_op>(parent)) {
        return emitOpError("yield must be inside if or for");
    }

    return mlir::success();
}

}
