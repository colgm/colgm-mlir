#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/if_op.hpp"

namespace colgm_mlir {

void if_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value condition, mlir::Type result_type) {
    state.addOperands(condition);
    if (result_type) {
        state.addTypes(result_type);
    }

    auto then_body = state.addRegion();
    then_body->emplaceBlock();

    auto else_body = state.addRegion();
    else_body->emplaceBlock();
}

if_op if_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                    mlir::Value condition, mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, condition, result_type);
    return llvm::cast<if_op>(builder.create(state));
}

mlir::ParseResult if_op::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand condition;

    // colgm.if %cond
    if (parser.parseOperand(condition) ||
        parser.resolveOperand(condition, parser.getBuilder().getI1Type(), result.operands)) {
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
    if (getNumResults() > 0) {
        p << " -> " << getResult(0).getType();
    }
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
        return emitOpError("condition must be i1, got ") << cond_type;
    }

    auto then_yield = llvm::cast<yield_op>(
        get_then_region().front().getTerminator());
    auto else_yield = llvm::cast<yield_op>(
        get_else_region().front().getTerminator());

    auto then_operands = then_yield.getNumOperands();
    auto else_operands = else_yield.getNumOperands();
    if (then_operands != else_operands) {
        return emitOpError("branches yield different number of values: ")
               << then_operands << " vs " << else_operands;
    }

    if (then_operands == 1) {
        if (getNumResults() != 1) {
            return emitOpError("yield has value but op has ") << getNumResults() << " results";
        }
        auto yield_type = then_yield.getOperand(0).getType();
        auto result_type = getResult(0).getType();
        if (yield_type != result_type) {
            return emitOpError("yield type ") << yield_type
                   << " does not match result type " << result_type;
        }
        if (else_yield.getOperand(0).getType() != result_type) {
            return emitOpError("else branch yield type ")
                   << else_yield.getOperand(0).getType()
                   << " does not match result type " << result_type;
        }
    } else {
        if (getNumResults() != 0) {
            return emitOpError("yield has no value but op has ") << getNumResults() << " results";
        }
    }

    return mlir::success();
}

}
