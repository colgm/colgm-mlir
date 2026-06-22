#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/for_op.hpp"

namespace colgm_mlir {

void for_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lower_bound, mlir::Value upper_bound) {
  state.addOperands({lower_bound, upper_bound});
  
  auto* body = state.addRegion();
  auto& block = body->emplaceBlock();
  block.addArgument(builder.getIndexType(), builder.getUnknownLoc());
}

for_op for_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value lower_bound, mlir::Value upper_bound) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lower_bound, upper_bound);
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

    return mlir::success();
}

}
