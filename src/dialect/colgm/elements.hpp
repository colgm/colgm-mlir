#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace colgm_mlir {

class elements_op: public mlir::Op<elements_op,
                                    mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "colgm.elements"; }
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    mlir::ArrayAttr get_target_shape() {
        return llvm::cast<mlir::ArrayAttr>((*this)->getAttr("target_shape"));
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::ValueRange values, mlir::ArrayRef<int64_t> shape);
    static elements_op create(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::ValueRange values, mlir::ArrayRef<int64_t> shape);
    static mlir::ParseResult parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result);
    void print(mlir::OpAsmPrinter& p);
    mlir::LogicalResult verify();
};

}
