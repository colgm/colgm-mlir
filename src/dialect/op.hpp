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
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

class relu_op: public mlir::Op<relu_op,
                            mlir::OpTrait::OneOperand,
                            mlir::OpTrait::OneResult,
                            mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.relu"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class neg_op: public mlir::Op<neg_op,
                           mlir::OpTrait::OneOperand,
                           mlir::OpTrait::OneResult,
                           mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.neg"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class abs_op: public mlir::Op<abs_op,
                           mlir::OpTrait::OneOperand,
                           mlir::OpTrait::OneResult,
                           mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.abs"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class exp_op: public mlir::Op<exp_op,
                           mlir::OpTrait::OneOperand,
                           mlir::OpTrait::OneResult,
                           mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.exp"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class log_op: public mlir::Op<log_op,
                           mlir::OpTrait::OneOperand,
                           mlir::OpTrait::OneResult,
                           mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.log"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class sqrt_op: public mlir::Op<sqrt_op,
                            mlir::OpTrait::OneOperand,
                            mlir::OpTrait::OneResult,
                            mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.sqrt"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class sigmoid_op: public mlir::Op<sigmoid_op,
                               mlir::OpTrait::OneOperand,
                               mlir::OpTrait::OneResult,
                               mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.sigmoid"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

class tanh_op: public mlir::Op<tanh_op,
                            mlir::OpTrait::OneOperand,
                            mlir::OpTrait::OneResult,
                            mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.tanh"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }
};

}