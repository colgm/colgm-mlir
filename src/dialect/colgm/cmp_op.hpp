#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace colgm_mlir {

class cmp_eq_op: public mlir::Op<cmp_eq_op,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.cmp_eq"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static cmp_eq_op create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

class cmp_ne_op: public mlir::Op<cmp_ne_op,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.cmp_ne"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static cmp_ne_op create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

class cmp_lt_op: public mlir::Op<cmp_lt_op,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.cmp_lt"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static cmp_lt_op create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

class cmp_le_op: public mlir::Op<cmp_le_op,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.cmp_le"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static cmp_le_op create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

class cmp_gt_op: public mlir::Op<cmp_gt_op,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.cmp_gt"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static cmp_gt_op create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

class cmp_ge_op: public mlir::Op<cmp_ge_op,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.cmp_ge"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lhs() { return getOperand(0); }
    mlir::Value get_rhs() { return getOperand(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static cmp_ge_op create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs, mlir::Type result_type);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
