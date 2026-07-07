#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/unary_op.hpp"

namespace colgm_mlir {

static bool is_float_type(mlir::Type type) {
    if (llvm::isa<mlir::FloatType>(type)) {
        return true;
    }

    if (auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
        return llvm::isa<mlir::FloatType>(tensor.getElementType());
    }
    return false;
}

void relu_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

relu_op relu_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<relu_op>(builder.create(state));
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

neg_op neg_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<neg_op>(builder.create(state));
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

abs_op abs_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<abs_op>(builder.create(state));
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

exp_op exp_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<exp_op>(builder.create(state));
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

mlir::LogicalResult exp_op::verify() {
    auto type = get_input().getType();
    if (!is_float_type(type)) {
        return mlir::emitError(getLoc(), "exp input must be a float type");
    }
    return mlir::success();
}

void log_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

log_op log_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<log_op>(builder.create(state));
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

mlir::LogicalResult log_op::verify() {
    auto type = get_input().getType();
    if (!is_float_type(type)) {
        return mlir::emitError(getLoc(), "log input must be a float type");
    }
    return mlir::success();
}

void sqrt_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

sqrt_op sqrt_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<sqrt_op>(builder.create(state));
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

mlir::LogicalResult sqrt_op::verify() {
    auto type = get_input().getType();
    if (!is_float_type(type)) {
        return mlir::emitError(getLoc(), "sqrt input must be a float type");
    }
    return mlir::success();
}

void sigmoid_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                       mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

sigmoid_op sigmoid_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<sigmoid_op>(builder.create(state));
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

mlir::LogicalResult sigmoid_op::verify() {
    auto type = get_input().getType();
    if (!is_float_type(type)) {
        return mlir::emitError(getLoc(), "sigmoid input must be a float type");
    }
    return mlir::success();
}

void tanh_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

tanh_op tanh_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value input) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input);
    return llvm::cast<tanh_op>(builder.create(state));
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

mlir::LogicalResult tanh_op::verify() {
    auto type = get_input().getType();
    if (!is_float_type(type)) {
        return mlir::emitError(getLoc(), "tanh input must be a float type");
    }
    return mlir::success();
}

}
