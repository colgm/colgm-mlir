#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/slice.hpp"

namespace colgm_mlir {

void slice_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                     mlir::Value input, mlir::Value index, i64 axis) {
    state.addOperands({input, index});

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto input_shape = input_type.getShape();

    llvm::SmallVector<i64> output_shape;
    for (i64 i = 0; i < static_cast<i64>(input_shape.size()); ++i) {
        if (i != axis)
            output_shape.push_back(input_shape[i]);
    }

    auto res_type = mlir::RankedTensorType::get(output_shape,
                                                input_type.getElementType());
    state.addTypes(res_type);
    state.addAttribute("axis", builder.getI64IntegerAttr(axis));
}

slice_op slice_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value input, mlir::Value index, i64 axis) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input, index, axis);
    return llvm::cast<slice_op>(builder.create(state));
}

mlir::ParseResult slice_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input, index;
    mlir::Type input_type, index_type, result_type;

    // %t, %idx {axis = N} : input_type, index_type -> result_type
    if (parser.parseOperand(input) ||
        parser.parseComma() ||
        parser.parseOperand(index) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(input_type) ||
        parser.parseComma() ||
        parser.parseType(index_type) ||
        parser.parseArrow() ||
        parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(input, input_type, result.operands) ||
        parser.resolveOperand(index, index_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void slice_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << ", " << get_index()
      << " {axis = " << get_axis() << "}"
      << " : " << get_input().getType()
      << ", " << get_index().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult slice_op::verify() {
    auto input_type = get_input().getType();
    if (!llvm::isa<mlir::RankedTensorType>(input_type)) {
        return emitOpError("input must be a ranked tensor");
    }
    auto input_rt = llvm::cast<mlir::RankedTensorType>(input_type);
    auto rank = input_rt.getRank();
    if (rank < 1) {
        return emitOpError("input rank must be >= 1");
    }

    auto index_type = get_index().getType();
    if (auto idx_tensor = llvm::dyn_cast<mlir::RankedTensorType>(index_type)) {
        if (idx_tensor.getRank() != 0) {
            return emitOpError("index must be a scalar (rank-0 tensor), got rank ")
                   << idx_tensor.getRank();
        }
        auto et = idx_tensor.getElementType();
        if (!llvm::isa<mlir::IntegerType>(et)) {
            return emitOpError("index tensor element type must be integer, got ")
                   << et;
        }
    } else if (!llvm::isa<mlir::IndexType>(index_type)) {
        return emitOpError("index must be of type index or tensor<i32/i64>, got ")
               << index_type;
    }

    if (!(*this)->hasAttr("axis")) {
        return emitOpError("missing 'axis' attribute");
    }

    auto axis_attr = llvm::cast<mlir::IntegerAttr>((*this)->getAttr("axis"));
    auto axis = axis_attr.getInt();
    if (axis < 0 || axis >= rank) {
        return emitOpError("axis ") << axis << " out of range [0, " << rank << ")";
    }

    auto output_type = (*this)->getResult(0).getType();
    if (!llvm::isa<mlir::RankedTensorType>(output_type)) {
        return emitOpError("output must be a ranked tensor");
    }
    auto output_rt = llvm::cast<mlir::RankedTensorType>(output_type);

    if (input_rt.getElementType() != output_rt.getElementType()) {
        return emitOpError("element type mismatch: input=")
               << input_rt.getElementType() << ", output="
               << output_rt.getElementType();
    }

    if (static_cast<i64>(output_rt.getRank()) != rank - 1) {
        return emitOpError("output rank ") << output_rt.getRank()
               << " != input rank " << rank << " - 1";
    }

    for (i64 i = 0, j = 0; i < rank; ++i) {
        if (i == axis) continue;
        if (input_rt.getDimSize(i) != output_rt.getDimSize(j)) {
            return emitOpError("output dim ") << j << " = "
                   << output_rt.getDimSize(j)
                   << " != input dim " << i << " = "
                   << input_rt.getDimSize(i);
        }
        ++j;
    }

    return mlir::success();
}

}
