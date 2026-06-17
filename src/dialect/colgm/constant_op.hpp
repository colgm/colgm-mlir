#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "utils/type.hpp"

namespace colgm_mlir {

class constant_op: public mlir::Op<constant_op,
                                   mlir::OpTrait::ZeroOperands,
                                   mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.constant"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::TypedAttr get_value() {
        return llvm::cast<mlir::TypedAttr>((*this)->getAttr("value"));
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::TypedAttr value, mlir::Type type);
    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      i64 value, mlir::Type type);
    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      f64 value, mlir::Type type);
    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      bool value, mlir::Type type);

    static constant_op create(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::TypedAttr value, mlir::Type type);
    static constant_op create(mlir::OpBuilder& builder, mlir::Location loc,
                              i64 value, mlir::Type type);
    static constant_op create(mlir::OpBuilder& builder, mlir::Location loc,
                              f64 value, mlir::Type type);
    static constant_op create(mlir::OpBuilder& builder, mlir::Location loc,
                              bool value, mlir::Type type);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
