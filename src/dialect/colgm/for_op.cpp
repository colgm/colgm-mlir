#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/for_op.hpp"

namespace colgm_mlir {

void for_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lower_bound, mlir::Value upper_bound,
                   mlir::Type result_type) {
    state.addOperands({lower_bound, upper_bound});
    if (result_type) {
        state.addTypes(result_type);
    }

    auto* body = state.addRegion();
    auto& block = body->emplaceBlock();
    block.addArgument(builder.getIndexType(), builder.getUnknownLoc());
}

for_op for_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value lower_bound, mlir::Value upper_bound,
                      mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lower_bound, upper_bound, result_type);
    return llvm::cast<for_op>(builder.create(state));
}

mlir::ParseResult for_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lower_bound, upper_bound;
    mlir::Type idx_type = parser.getBuilder().getIndexType();

    // colgm.for %i = %lower_bound to %upper_bound
    mlir::OpAsmParser::Argument induction_var;
    if (parser.parseArgument(induction_var, false, false) ||
        parser.parseEqual() ||
        parser.parseOperand(lower_bound) ||
        parser.parseKeyword("to") ||
        parser.parseOperand(upper_bound)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lower_bound, idx_type, result.operands) ||
        parser.resolveOperand(upper_bound, idx_type, result.operands)) {
        return mlir::failure();
    }

    // optional -> result_type
    if (mlir::succeeded(parser.parseOptionalArrow())) {
        mlir::Type result_type;
        if (parser.parseType(result_type)) {
            return mlir::failure();
        }
        result.addTypes(result_type);
    }

    // { body region }
    auto body = result.addRegion();
    induction_var.type = idx_type;
    if (parser.parseRegion(*body, {induction_var})) {
        return mlir::failure();
    }

    return mlir::success();
}

void for_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_induction_var() << " = "
      << get_lower_bound() << " to " << get_upper_bound();
    if (getNumResults() > 0) {
        p << " -> " << getResult(0).getType();
    }
    p << " ";
    p.printRegion(get_body(), false, true);
}

mlir::LogicalResult for_op::verify() {
    auto lower_bound = get_lower_bound();
    auto upper_bound = get_upper_bound();
    auto lb_type = lower_bound.getType();
    auto ub_type = upper_bound.getType();
    auto idx_type = mlir::IndexType::get(getContext());

    if (lb_type != idx_type) {
        return emitOpError("lower bound must be index, got ") << lb_type;
    }
    if (ub_type != idx_type) {
        return emitOpError("upper bound must be index, got ") << ub_type;
    }
    if (get_body().getBlocks().front().getNumArguments() != 1) {
        return emitOpError("body must have exactly one argument");
    }
    if (get_induction_var().getType() != idx_type) {
        return emitOpError("induction variable must be index, got ")
               << get_induction_var().getType();
    }

    auto yield_op = llvm::cast<colgm_mlir::yield_op>(
        get_body().getBlocks().front().getTerminator());
    auto yield_operands = yield_op.getNumOperands();

    if (yield_operands == 1) {
        if (getNumResults() != 1) {
            return emitOpError("yield has value but op has ") << getNumResults() << " results";
        }
        if (yield_op.getOperand(0).getType() != getResult(0).getType()) {
            return emitOpError("yield type ")
                   << yield_op.getOperand(0).getType()
                   << " does not match result type " << getResult(0).getType();
        }
    } else {
        if (getNumResults() != 0) {
            return emitOpError("yield has no value but op has ") << getNumResults() << " results";
        }
    }

    return mlir::success();
}

}
