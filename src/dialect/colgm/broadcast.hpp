#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "utils/type.hpp"

namespace colgm_mlir {

class broadcast_op: public mlir::Op<broadcast_op,
                                    mlir::OpTrait::OneOperand,
                                    mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.broadcast"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    mlir::ArrayAttr get_target_shape() {
        return llvm::cast<mlir::ArrayAttr>((*this)->getAttr("target_shape"));
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input, mlir::ArrayRef<i64> shape);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
