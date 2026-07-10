#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace colgm_mlir {

class print_op: public mlir::Op<print_op,
                                mlir::OpTrait::VariadicOperands,
                                mlir::OpTrait::ZeroResults> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.print"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::OperandRange get_inputs() { return getOperands(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::ValueRange inputs);
    static print_op create(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::ValueRange inputs);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify() { return mlir::success(); }

    void getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
        effects.emplace_back(mlir::MemoryEffects::Write::get());
    }
};

}
