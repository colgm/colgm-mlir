#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dialect/colgm/yield.hpp"

namespace colgm_mlir {

class if_op: public mlir::Op<if_op,
                             mlir::OpTrait::OneOperand,
                             mlir::OpTrait::VariadicResults,
                             mlir::OpTrait::SingleBlockImplicitTerminator<yield_op>::Impl> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.if"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::Value get_condition() { return getOperand(); }
    mlir::Region& get_then_region() { return (*this)->getRegion(0); }
    mlir::Region& get_else_region() { return (*this)->getRegion(1); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value condition,
                      mlir::Type result_type = {});
    static if_op create(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value condition,
                        mlir::Type result_type = {});
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
