#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace colgm_mlir {

class yield_op: public mlir::Op<yield_op,
                                mlir::OpTrait::IsTerminator,
                                mlir::OpTrait::ZeroOperands,
                                mlir::OpTrait::ZeroResults> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.yield"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    static void build(mlir::OpBuilder&, mlir::OperationState&) {}
    static yield_op create(mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::OperationState state(loc, getOperationName());
        build(builder, state);
        return llvm::cast<yield_op>(builder.create(state));
    }

    static mlir::ParseResult parse(mlir::OpAsmParser&, mlir::OperationState&) {
        return mlir::success();
    }
    void print(mlir::OpAsmPrinter&) {}
    mlir::LogicalResult verify();
};

}
