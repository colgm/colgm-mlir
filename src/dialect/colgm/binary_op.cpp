#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/binary_op.hpp"

namespace colgm_mlir {

void add_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

add_op add_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value lhs, mlir::Value rhs) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs);
    return llvm::cast<add_op>(builder.create(state));
}

mlir::ParseResult add_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void add_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void sub_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

sub_op sub_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value lhs, mlir::Value rhs) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs);
    return llvm::cast<sub_op>(builder.create(state));
}

mlir::ParseResult sub_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void sub_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void mul_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

mul_op mul_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value lhs, mlir::Value rhs) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs);
    return llvm::cast<mul_op>(builder.create(state));
}

mlir::ParseResult mul_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void mul_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void div_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

div_op div_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::Value lhs, mlir::Value rhs) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs);
    return llvm::cast<div_op>(builder.create(state));
}

mlir::ParseResult div_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void div_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void matmul_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = llvm::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = llvm::cast<mlir::RankedTensorType>(rhs.getType());

    llvm::SmallVector<i64> shape;
    for (usize i = 0; i < lhs_type.getRank() - 2; i++) {
        shape.push_back(lhs_type.getDimSize(i));
    }
    shape.push_back(lhs_type.getDimSize(lhs_type.getRank() - 2));
    shape.push_back(rhs_type.getDimSize(rhs_type.getRank() - 1));

    auto res_type = mlir::RankedTensorType::get(shape, lhs_type.getElementType());
    state.addOperands({lhs, rhs});
    state.addTypes(res_type);
}

matmul_op matmul_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs);
    return llvm::cast<matmul_op>(builder.create(state));
}

mlir::ParseResult matmul_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type lhs_type, rhs_type;

    // %lhs, %rhs : type, type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(lhs_type) ||
        parser.parseComma() || parser.parseType(rhs_type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, lhs_type, result.operands) ||
        parser.resolveOperand(rhs, rhs_type, result.operands)) {
        return mlir::failure();
    }

    auto lhs_tensor = llvm::cast<mlir::RankedTensorType>(lhs_type);
    auto rhs_tensor = llvm::cast<mlir::RankedTensorType>(rhs_type);

    if (lhs_tensor.getRank() != rhs_tensor.getRank()) {
        return parser.emitError(parser.getCurrentLocation(), "tensor rank must be equal");
    }
    if (lhs_tensor.getRank() < 2 || rhs_tensor.getRank() < 2) {
        return parser.emitError(parser.getCurrentLocation(), "tensor rank must be >= 2");
    }

    llvm::SmallVector<i64> shape;
    for (usize i = 0; i < lhs_tensor.getRank() - 2; i++) {
        shape.push_back(lhs_tensor.getDimSize(i));
    }
    shape.push_back(lhs_tensor.getDimSize(lhs_tensor.getRank() - 2));
    shape.push_back(rhs_tensor.getDimSize(rhs_tensor.getRank() - 1));

    auto res_type = mlir::RankedTensorType::get(shape, lhs_tensor.getElementType());
    result.addTypes(res_type);
    return mlir::success();
}

void matmul_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType() << ", " << get_rhs().getType();
}

mlir::LogicalResult matmul_op::verify() {
    auto lhs_type = llvm::cast<mlir::RankedTensorType>(get_lhs().getType());
    auto rhs_type = llvm::cast<mlir::RankedTensorType>(get_rhs().getType());
    if (lhs_type.getRank() < 2 || rhs_type.getRank() < 2) {
        return emitOpError("tensor rank must be >= 2");
    }
    if (lhs_type.getRank() != rhs_type.getRank()) {
        return emitOpError("tensor rank must be equal");
    }

    for (usize i = 0; i < lhs_type.getRank() - 2; i++) {
        if (lhs_type.getDimSize(i) != rhs_type.getDimSize(i)) {
            return emitOpError("lhs[") << i << "] must equal rhs[" << i << "]";
        }
    }

    if (lhs_type.getDimSize(lhs_type.getRank() - 1) != rhs_type.getDimSize(rhs_type.getRank() - 2)) {
        return emitOpError("lhs[-1] must equal rhs[-2]");
    }
    return mlir::success();
}

}
