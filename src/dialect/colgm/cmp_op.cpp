#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/cmp_op.hpp"

namespace colgm_mlir {

static mlir::LogicalResult verify_cmp(mlir::Operation* op) {
    auto lhs_type = llvm::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto rhs_type = llvm::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto res_type = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    if (lhs_type != rhs_type) {
        return op->emitOpError("operand types must match: ")
               << lhs_type << " vs " << rhs_type;
    }

    if (!res_type.getElementType().isInteger(1)) {
        return op->emitOpError("result element type must be i1, got ")
               << res_type.getElementType();
    }

    if (res_type.getShape() != lhs_type.getShape()) {
        return op->emitOpError("result shape must match operand shape: ")
               << res_type.getShape() << " vs " << lhs_type.getShape();
    }

    return mlir::success();
}

void cmp_eq_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    state.addOperands({lhs, rhs});
    state.addTypes(result_type);
}

cmp_eq_op cmp_eq_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs, result_type);
    return llvm::cast<cmp_eq_op>(builder.create(state));
}

mlir::ParseResult cmp_eq_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type operand_type, result_type;

    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(operand_type) ||
        parser.parseArrow() || parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(lhs, operand_type, result.operands) ||
        parser.resolveOperand(rhs, operand_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void cmp_eq_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult cmp_eq_op::verify() { return verify_cmp(*this); }

void cmp_ne_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    state.addOperands({lhs, rhs});
    state.addTypes(result_type);
}

cmp_ne_op cmp_ne_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs, result_type);
    return llvm::cast<cmp_ne_op>(builder.create(state));
}

mlir::ParseResult cmp_ne_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type operand_type, result_type;

    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(operand_type) ||
        parser.parseArrow() || parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(lhs, operand_type, result.operands) ||
        parser.resolveOperand(rhs, operand_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void cmp_ne_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult cmp_ne_op::verify() { return verify_cmp(*this); }

void cmp_lt_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    state.addOperands({lhs, rhs});
    state.addTypes(result_type);
}

cmp_lt_op cmp_lt_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs, result_type);
    return llvm::cast<cmp_lt_op>(builder.create(state));
}

mlir::ParseResult cmp_lt_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type operand_type, result_type;

    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(operand_type) ||
        parser.parseArrow() || parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(lhs, operand_type, result.operands) ||
        parser.resolveOperand(rhs, operand_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void cmp_lt_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult cmp_lt_op::verify() { return verify_cmp(*this); }

void cmp_le_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    state.addOperands({lhs, rhs});
    state.addTypes(result_type);
}

cmp_le_op cmp_le_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs, result_type);
    return llvm::cast<cmp_le_op>(builder.create(state));
}

mlir::ParseResult cmp_le_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type operand_type, result_type;

    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(operand_type) ||
        parser.parseArrow() || parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(lhs, operand_type, result.operands) ||
        parser.resolveOperand(rhs, operand_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void cmp_le_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult cmp_le_op::verify() { return verify_cmp(*this); }

void cmp_gt_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    state.addOperands({lhs, rhs});
    state.addTypes(result_type);
}

cmp_gt_op cmp_gt_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs, result_type);
    return llvm::cast<cmp_gt_op>(builder.create(state));
}

mlir::ParseResult cmp_gt_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type operand_type, result_type;

    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(operand_type) ||
        parser.parseArrow() || parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(lhs, operand_type, result.operands) ||
        parser.resolveOperand(rhs, operand_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void cmp_gt_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult cmp_gt_op::verify() { return verify_cmp(*this); }

void cmp_ge_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    state.addOperands({lhs, rhs});
    state.addTypes(result_type);
}

cmp_ge_op cmp_ge_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, lhs, rhs, result_type);
    return llvm::cast<cmp_ge_op>(builder.create(state));
}

mlir::ParseResult cmp_ge_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type operand_type, result_type;

    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(operand_type) ||
        parser.parseArrow() || parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(lhs, operand_type, result.operands) ||
        parser.resolveOperand(rhs, operand_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void cmp_ge_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult cmp_ge_op::verify() { return verify_cmp(*this); }

}
