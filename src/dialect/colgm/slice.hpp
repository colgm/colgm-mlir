#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "utils/type.hpp"

namespace colgm_mlir {

class slice_op: public mlir::Op<slice_op,
                                mlir::OpTrait::NOperands<2>::Impl,
                                mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.slice"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(0); }
    mlir::Value get_index() { return getOperand(1); }

    i64 get_axis() {
        return llvm::cast<mlir::IntegerAttr>((*this)->getAttr("axis")).getInt();
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input, mlir::Value index, i64 axis);
    static slice_op create(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value input, mlir::Value index, i64 axis);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
