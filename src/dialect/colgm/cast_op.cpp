#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/cast_op.hpp"

namespace colgm_mlir {

static bool is_valid_cast(mlir::Type src, mlir::Type dst) {
    if (src == dst)
        return true;
    if (llvm::isa<mlir::IntegerType>(src) && llvm::isa<mlir::IntegerType>(dst))
        return true;
    if (llvm::isa<mlir::FloatType>(src) && llvm::isa<mlir::FloatType>(dst))
        return true;
    if (llvm::isa<mlir::IntegerType>(src) && llvm::isa<mlir::IndexType>(dst))
        return true;
    if (llvm::isa<mlir::IndexType>(src) && llvm::isa<mlir::IntegerType>(dst))
        return true;
    return false;
}

void cast_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input, mlir::Type result_type) {
    state.addOperands({input});
    state.addTypes(result_type);
}

mlir::ParseResult cast_op::parse(mlir::OpAsmParser& parser,
                                  mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type input_type, result_type;

    if (parser.parseOperand(input) ||
        parser.parseColonType(input_type) ||
        parser.parseArrow() ||
        parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(input, input_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void cast_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult cast_op::verify() {
    auto src = get_input().getType();
    auto dst = (*this)->getResult(0).getType();

    auto src_tensor = llvm::dyn_cast<mlir::RankedTensorType>(src);
    auto dst_tensor = llvm::dyn_cast<mlir::RankedTensorType>(dst);

    if (src_tensor && dst_tensor) {
        if (src_tensor.getShape() != dst_tensor.getShape()) {
            return emitOpError("tensor shape mismatch: ")
                   << src_tensor.getShape() << " vs " << dst_tensor.getShape();
        }
        if (!is_valid_cast(src_tensor.getElementType(),
                           dst_tensor.getElementType())) {
            return emitOpError("cannot cast tensor element type from ")
                   << src_tensor.getElementType() << " to "
                   << dst_tensor.getElementType();
        }
        return mlir::success();
    }

    if (!src_tensor && !dst_tensor) {
        if (!is_valid_cast(src, dst)) {
            return emitOpError("cannot cast from ") << src << " to " << dst;
        }
        return mlir::success();
    }

    return emitOpError("cannot cast between scalar and tensor type");
}

}
