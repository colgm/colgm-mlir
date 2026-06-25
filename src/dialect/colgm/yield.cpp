#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/yield.hpp"
#include "dialect/colgm/if_op.hpp"
#include "dialect/colgm/for_op.hpp"

namespace colgm_mlir {

void yield_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                     mlir::ValueRange values) {
    state.addOperands(values);
}

yield_op yield_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::ValueRange values) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, values);
    return llvm::cast<yield_op>(builder.create(state));
}

mlir::ParseResult yield_op::parse(mlir::OpAsmParser& parser,
                                  mlir::OperationState& result) {
    // colgm.yield [%v1, %v2, ... : type1, type2, ...]
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
    llvm::SmallVector<mlir::Type, 4> types;

    // Parse optional operand list
    mlir::OpAsmParser::UnresolvedOperand first;
    if (parser.parseOptionalOperand(first).has_value()) {
        operands.push_back(first);
        // parse rest: , %v2, %v3, ...
        while (mlir::succeeded(parser.parseOptionalComma())) {
            mlir::OpAsmParser::UnresolvedOperand next;
            if (parser.parseOperand(next)) {
                return mlir::failure();
            }
            operands.push_back(next);
        }

        if (parser.parseColon()) {
            return mlir::failure();
        }

        // parse type list
        if (parser.parseType(types.emplace_back())) {
            return mlir::failure();
        }
        while (mlir::succeeded(parser.parseOptionalComma())) {
            if (parser.parseType(types.emplace_back())) {
                return mlir::failure();
            }
        }

        if (operands.size() != types.size()) {
            return parser.emitError(parser.getCurrentLocation(),
                                    "operand count does not match type count");
        }
        for (usize i = 0; i < operands.size(); ++i) {
            if (parser.resolveOperand(operands[i], types[i], result.operands)) {
                return mlir::failure();
            }
        }
    }

    return mlir::success();
}

void yield_op::print(mlir::OpAsmPrinter& p) {
    if (getNumOperands() > 0) {
        p << " ";
        p.printOperands(getOperands());
        p << " : ";
        llvm::interleaveComma(getOperands().getTypes(), p);
    }
}

mlir::LogicalResult yield_op::verify() {
    auto parent = (*this)->getParentOp();
    if (!llvm::isa<if_op>(parent) && !llvm::isa<for_op>(parent)) {
        return emitOpError("yield must be inside if or for");
    }

    // if_op allows at most 1 operand
    if (llvm::isa<if_op>(parent) && getNumOperands() > 1) {
        return emitOpError("yield inside if can have at most one operand");
    }

    return mlir::success();
}

}
