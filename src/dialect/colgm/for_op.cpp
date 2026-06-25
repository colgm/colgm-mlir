#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/for_op.hpp"

namespace colgm_mlir {

void for_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lower_bound, mlir::Value upper_bound,
                   mlir::ValueRange iter_args,
                   mlir::TypeRange result_types) {
    state.addOperands(lower_bound);
    state.addOperands(upper_bound);
    state.addOperands(iter_args);
    state.addTypes(result_types);

    auto* body = state.addRegion();
    auto& block = body->emplaceBlock();
    block.addArgument(builder.getIndexType(), builder.getUnknownLoc());
    for (auto t : result_types) {
        block.addArgument(t, builder.getUnknownLoc());
    }
}

for_op for_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value lower_bound, mlir::Value upper_bound,
                      mlir::ValueRange iter_args,
                      mlir::TypeRange result_types) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lower_bound, upper_bound, iter_args, result_types);
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

    induction_var.type = idx_type;

    // optional iter_args(%name = %init, ...)
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> iter_operands;
    llvm::SmallVector<mlir::OpAsmParser::Argument, 4> iter_args;
    if (mlir::succeeded(parser.parseOptionalKeyword("iter_args"))) {
        if (parser.parseLParen()) {
            return mlir::failure();
        }
        while (!mlir::succeeded(parser.parseOptionalRParen())) {
            mlir::OpAsmParser::Argument arg;
            mlir::OpAsmParser::UnresolvedOperand init;
            if (parser.parseArgument(arg, false, false) ||
                parser.parseEqual() ||
                parser.parseOperand(init)) {
                return mlir::failure();
            }
            iter_args.push_back(arg);
            iter_operands.push_back(init);
            if (mlir::succeeded(parser.parseOptionalRParen())) {
                break;
            }
            if (parser.parseComma()) {
                return mlir::failure();
            }
        }
    }

    // optional -> (result_type1, result_type2, ...)  or  -> single_type
    bool has_arrow = mlir::succeeded(parser.parseOptionalArrow());
    if (has_arrow) {
        auto lparen_loc = parser.getCurrentLocation();
        if (mlir::succeeded(parser.parseOptionalLParen())) {
            // multiple types: -> (type1, type2, ...)
            llvm::SmallVector<mlir::Type> types;
            do {
                mlir::Type t;
                if (parser.parseType(t)) {
                    return mlir::failure();
                }
                types.push_back(t);
            } while (mlir::succeeded(parser.parseOptionalComma()));
            if (parser.parseRParen()) {
                return mlir::failure();
            }
            for (auto t : types) result.addTypes(t);
        } else {
            // single type (backward-compatible)
            mlir::Type single_type;
            if (parser.parseType(single_type)) {
                return mlir::failure();
            }
            result.addTypes(single_type);
        }

        // set iter_args types from result types
        for (usize i = 0; i < iter_args.size() && i < result.types.size(); ++i) {
            iter_args[i].type = result.types[i];
        }
    }

    // resolve iter_args init operands
    for (usize i = 0; i < iter_operands.size(); ++i) {
        if (parser.resolveOperand(iter_operands[i], iter_args[i].type,
                                  result.operands)) {
            return mlir::failure();
        }
    }

    // body region
    auto body = result.addRegion();
    llvm::SmallVector<mlir::OpAsmParser::Argument, 4> block_args = { induction_var };
    for (auto& arg : iter_args) {
        block_args.push_back(arg);
    }
    if (parser.parseRegion(*body, block_args)) {
        return mlir::failure();
    }

    return mlir::success();
}

void for_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_induction_var() << " = "
      << get_lower_bound() << " to " << get_upper_bound();

    auto iter_args = get_iter_args();
    if (!iter_args.empty()) {
        p << " iter_args(";
        auto& block = get_body().getBlocks().front();
        for (usize i = 0; i < iter_args.size(); ++i) {
            if (i > 0) p << ", ";
            p << block.getArgument(1 + i) << " = " << iter_args[i];
        }
        p << ")";
    }

    if (getNumResults() > 0) {
        p << " -> ";
        if (getNumResults() == 1) {
            p << getResult(0).getType();
        } else {
            p << "(";
            llvm::interleaveComma(getResultTypes(), p);
            p << ")";
        }
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

    auto num_iter_args = getNumIterArgs();
    auto num_results = getNumResults();
    auto& block = get_body().getBlocks().front();

    // block args: induction var + iter_args
    if (block.getNumArguments() != 1 + num_iter_args) {
        return emitOpError("body must have ")
               << (1 + num_iter_args) << " arguments (induction var"
               << (num_iter_args > 0 ? " + iter_args" : "") << ")";
    }

    if (get_induction_var().getType() != idx_type) {
        return emitOpError("induction variable must be index, got ")
               << get_induction_var().getType();
    }

    // iter_args types must match result types
    for (unsigned i = 0; i < num_iter_args; ++i) {
        auto iter_arg_type = block.getArgument(1 + i).getType();
        if (iter_arg_type != getResult(i).getType()) {
            return emitOpError("iter_arg #") << i << " type "
                   << iter_arg_type << " does not match result type "
                   << getResult(i).getType();
        }
    }

    // results count must equal iter_args count
    if (num_results != num_iter_args) {
        return emitOpError("result count (") << num_results
               << ") must equal iter_args count (" << num_iter_args << ")";
    }

    // yield operands must match result count and types
    auto yield_op = llvm::cast<colgm_mlir::yield_op>(block.getTerminator());
    if (yield_op.getNumOperands() != num_results) {
        return emitOpError("yield has ") << yield_op.getNumOperands()
               << " operands but op has " << num_results << " results";
    }

    for (unsigned i = 0; i < num_results; ++i) {
        if (yield_op.getOperand(i).getType() != getResult(i).getType()) {
            return emitOpError("yield operand #") << i << " type "
                   << yield_op.getOperand(i).getType()
                   << " does not match result type "
                   << getResult(i).getType();
        }
    }

    return mlir::success();
}

}
