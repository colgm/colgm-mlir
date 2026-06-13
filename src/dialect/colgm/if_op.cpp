#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/if_op.hpp"

namespace colgm_mlir {

void if_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value condition) {
    state.addOperands(condition);

    auto then_body = state.addRegion();
    then_body->emplaceBlock();

    auto else_body = state.addRegion();
    else_body->emplaceBlock();
}

mlir::ParseResult if_op::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand condition;

    // colgm.if %cond
    if (parser.parseOperand(condition) ||
        parser.resolveOperand(condition, parser.getBuilder().getI1Type(), result.operands)) {
        return mlir::failure();
    }

    // { then region }
    auto then_region = result.addRegion();
    if (parser.parseRegion(*then_region, {}, {})) {
        return mlir::failure();
    }
    if_op::ensureTerminator(*then_region, parser.getBuilder(), result.location);

    // else { else region }
    auto else_region = result.addRegion();
    if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
        if (parser.parseRegion(*else_region, {}, {})) {
            return mlir::failure();
        }
    } else {
        else_region->emplaceBlock();
    }
    if_op::ensureTerminator(*else_region, parser.getBuilder(), result.location);

    return mlir::success();
}

void if_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_condition();
    p << " ";
    p.printRegion(get_then_region(), false);

    auto& else_region = get_else_region();
    if (!else_region.empty()) {
        p << " else ";
        p.printRegion(else_region, false);
    }
}

mlir::LogicalResult if_op::verify() {
    auto cond_type = get_condition().getType();
    if (cond_type != mlir::IntegerType::get(getContext(), 1)) {
        return emitOpError("colgm.if condition must be i1, got ") << cond_type;
    }
    return mlir::success();
}

}
