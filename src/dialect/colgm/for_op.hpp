#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dialect/colgm/yield.hpp"

namespace colgm_mlir {

class for_op: public mlir::Op<for_op,
                              mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::ZeroResults,
                              mlir::OpTrait::SingleBlockImplicitTerminator<yield_op>::Impl> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.for"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_lower_bound() { return getOperand(0); }
    mlir::Value get_upper_bound() { return getOperand(1); }
    mlir::Region& get_body() { return (*this)->getRegion(0); }
    mlir::BlockArgument get_induction_var() {
        return get_body().getBlocks().front().getArgument(0);
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lower_bound, mlir::Value upper_bound);
    static for_op create(mlir::OpBuilder& builder, mlir::Location loc,
                         mlir::Value lower_bound, mlir::Value upper_bound);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}