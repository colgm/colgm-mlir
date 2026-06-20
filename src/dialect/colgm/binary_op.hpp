#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace colgm_mlir {

class add_op: public mlir::Op<add_op,
                              mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult,
                              mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.add"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs);
    static add_op create(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class sub_op: public mlir::Op<sub_op,
                              mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult,
                              mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.sub"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs);
    static sub_op create(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class mul_op: public mlir::Op<mul_op,
                              mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult,
                              mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.mul"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs);
    static mul_op create(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class div_op: public mlir::Op<div_op,
                              mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult,
                              mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.div"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {};}

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs);
    static div_op create(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class matmul_op: public mlir::Op<matmul_op,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.matmul"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs);
    static matmul_op create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
