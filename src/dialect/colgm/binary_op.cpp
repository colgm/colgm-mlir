#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/binary_op.hpp"

namespace colgm_mlir {

void add_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
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
    auto elem_type = lhs_type.getElementType();
    auto res_type = mlir::RankedTensorType::get({
        lhs_type.getDimSize(0),
        rhs_type.getDimSize(1),
    }, elem_type);
    state.addOperands({lhs, rhs});
    state.addTypes(res_type);
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

    auto elem_type = llvm::cast<mlir::RankedTensorType>(lhs_type).getElementType();
    auto res_type = mlir::RankedTensorType::get({
        llvm::cast<mlir::RankedTensorType>(lhs_type).getDimSize(0),
        llvm::cast<mlir::RankedTensorType>(rhs_type).getDimSize(1),
    }, elem_type);
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
    if (lhs_type.getRank() != 2 || rhs_type.getRank() != 2) {
        return emitOpError("only support 2D tensor");
    }
    if (lhs_type.getDimSize(1) != rhs_type.getDimSize(0)) {
        return emitOpError("lhs cols must equal rhs rows");
    }
    return mlir::success();
}

}
