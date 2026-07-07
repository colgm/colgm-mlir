#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace colgm_mlir {

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
    static relu_op create(mlir::OpBuilder& builder, mlir::Location loc,
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
    static neg_op create(mlir::OpBuilder& builder, mlir::Location loc,
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
    static abs_op create(mlir::OpBuilder& builder, mlir::Location loc,
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
    static exp_op create(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
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
    static log_op create(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
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
    static sqrt_op create(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
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
    static sigmoid_op create(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
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
    static tanh_op create(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::Value input);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
