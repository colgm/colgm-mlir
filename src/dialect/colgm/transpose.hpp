#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "utils/type.hpp"

namespace colgm_mlir {

class transpose_op: public mlir::Op<transpose_op,
                                    mlir::OpTrait::OneOperand,
                                    mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.transpose"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_input() { return getOperand(); }

    mlir::ArrayAttr get_permutation() {
        return llvm::cast<mlir::ArrayAttr>((*this)->getAttr("permutation"));
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value input, mlir::ArrayRef<i64> permutation);
    static transpose_op create(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value input, mlir::ArrayRef<i64> permutation);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
