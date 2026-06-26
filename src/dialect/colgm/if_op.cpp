#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/if_op.hpp"

namespace colgm_mlir {

void if_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value condition, mlir::TypeRange result_types) {
    state.addOperands(condition);
    if (!result_types.empty()) {
        state.addTypes(result_types);
    }

    auto then_body = state.addRegion();
    then_body->emplaceBlock();

    auto else_body = state.addRegion();
    else_body->emplaceBlock();
}

if_op if_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                    mlir::Value condition, mlir::TypeRange result_types) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, condition, result_types);
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

    // optional -> result_type  or  -> (type1, type2, ...)
    llvm::SmallVector<mlir::Type> res_types;
    if (parser.parseOptionalArrowTypeList(res_types)) {
        return mlir::failure();
    }
    result.addTypes(res_types);

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
    auto num_results = getNumResults();
    if (num_results == 1) {
        p << " -> " << getResult(0).getType();
    } else if (num_results > 1) {
        p << " -> (" << getResult(0).getType();
        for (unsigned i = 1; i < num_results; ++i) {
            p << ", " << getResult(i).getType();
        }
        p << ")";
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

    auto num_results = getNumResults();
    if (then_operands != num_results) {
        return emitOpError("yield operand count ") << then_operands
               << " does not match result count " << num_results;
    }

    for (unsigned i = 0; i < num_results; ++i) {
        auto then_type = then_yield.getOperand(i).getType();
        auto else_type = else_yield.getOperand(i).getType();
        auto res_type = getResult(i).getType();
        if (then_type != res_type) {
            return emitOpError("then branch yield operand #") << i << " type "
                   << then_type << " does not match result type " << res_type;
        }
        if (else_type != res_type) {
            return emitOpError("else branch yield operand #") << i << " type "
                   << else_type << " does not match result type " << res_type;
        }
    }

    return mlir::success();
}

}
