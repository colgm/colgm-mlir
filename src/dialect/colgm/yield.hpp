#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace colgm_mlir {

class yield_op: public mlir::Op<yield_op,
                                mlir::OpTrait::IsTerminator,
                                mlir::OpTrait::VariadicOperands,
                                mlir::OpTrait::ZeroResults> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.yield"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value value = {});
    static yield_op create(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value value = {});

    static mlir::ParseResult parse(mlir::OpAsmParser&, mlir::OperationState&);
    void print(mlir::OpAsmPrinter&);
    mlir::LogicalResult verify();
};

}
