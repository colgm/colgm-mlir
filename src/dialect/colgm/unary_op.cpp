#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/unary_op.hpp"

namespace colgm_mlir {

void relu_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult relu_op::parse(mlir::OpAsmParser& parser,
                                 mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    // %input : type
    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void relu_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void neg_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult neg_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void neg_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void abs_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult abs_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void abs_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void exp_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult exp_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void exp_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void log_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult log_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void log_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void sqrt_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult sqrt_op::parse(mlir::OpAsmParser& parser,
                                 mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void sqrt_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void sigmoid_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                       mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult sigmoid_op::parse(mlir::OpAsmParser& parser,
                                    mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void sigmoid_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void tanh_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult tanh_op::parse(mlir::OpAsmParser& parser,
                                 mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void tanh_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

}
