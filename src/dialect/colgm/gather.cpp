#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/gather.hpp"

namespace colgm_mlir {

void gather_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value params, mlir::Value indices, i64 axis) {
    state.addOperands({params, indices});

    auto params_type = llvm::cast<mlir::RankedTensorType>(params.getType());
    auto indices_type = llvm::cast<mlir::RankedTensorType>(indices.getType());
    auto params_shape = params_type.getShape();
    auto indices_shape = indices_type.getShape();

    auto rank = static_cast<i64>(params_shape.size());
    auto indices_rank = static_cast<i64>(indices_shape.size());

    llvm::SmallVector<i64> output_shape;
    for (i64 i = 0; i < axis; ++i) {
        output_shape.push_back(params_shape[i]);
    }
    for (i64 i = 0; i < indices_rank; ++i) {
        output_shape.push_back(indices_shape[i]);
    }
    for (i64 i = axis + 1; i < rank; ++i) {
        output_shape.push_back(params_shape[i]);
    }

    auto res_type = mlir::RankedTensorType::get(output_shape,
                                                params_type.getElementType());
    state.addTypes(res_type);
    state.addAttribute("axis", builder.getI64IntegerAttr(axis));
}

gather_op gather_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value params, mlir::Value indices, i64 axis) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, params, indices, axis);
    return llvm::cast<gather_op>(builder.create(state));
}

mlir::ParseResult gather_op::parse(mlir::OpAsmParser& parser,
                                    mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand params, indices;
    mlir::Type params_type, indices_type, result_type;

    if (parser.parseOperand(params) ||
        parser.parseComma() ||
        parser.parseOperand(indices) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(params_type) ||
        parser.parseComma() ||
        parser.parseType(indices_type) ||
        parser.parseArrow() ||
        parser.parseType(result_type))
        return mlir::failure();

    if (parser.resolveOperand(params, params_type, result.operands) ||
        parser.resolveOperand(indices, indices_type, result.operands))
        return mlir::failure();

    result.addTypes(result_type);
    return mlir::success();
}

void gather_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_params()
      << ", " << get_indices()
      << " {axis = " << get_axis() << "}"
      << " : " << get_params().getType()
      << ", " << get_indices().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult gather_op::verify() {
    auto params_type = get_params().getType();
    if (!llvm::isa<mlir::RankedTensorType>(params_type)) {
        return emitOpError("params must be a ranked tensor");
    }
    auto params_rt = llvm::cast<mlir::RankedTensorType>(params_type);
    auto params_rank = static_cast<i64>(params_rt.getRank());

    auto indices_type = get_indices().getType();
    if (!llvm::isa<mlir::RankedTensorType>(indices_type)) {
        return emitOpError("indices must be a ranked tensor");
    }
    auto indices_rt = llvm::cast<mlir::RankedTensorType>(indices_type);
    auto indices_rank = static_cast<i64>(indices_rt.getRank());

    auto indices_elem = indices_rt.getElementType();
    if (!llvm::isa<mlir::IntegerType>(indices_elem)) {
        return emitOpError("indices element type must be integer, got ")
               << indices_elem;
    }

    if (!(*this)->hasAttr("axis")) {
        return emitOpError("missing 'axis' attribute");
    }

    auto axis_attr = llvm::cast<mlir::IntegerAttr>((*this)->getAttr("axis"));
    auto axis = axis_attr.getInt();
    if (axis < 0 || axis >= params_rank) {
        return emitOpError("axis ") << axis << " out of range [0, "
                   << params_rank << ")";
    }

    auto output_type = (*this)->getResult(0).getType();
    if (!llvm::isa<mlir::RankedTensorType>(output_type)) {
        return emitOpError("output must be a ranked tensor");
    }
    auto output_rt = llvm::cast<mlir::RankedTensorType>(output_type);

    if (params_rt.getElementType() != output_rt.getElementType()) {
        return emitOpError("element type mismatch: params=")
               << params_rt.getElementType() << ", output="
               << output_rt.getElementType();
    }

    auto expected_rank = params_rank + indices_rank - 1;
    if (static_cast<i64>(output_rt.getRank()) != expected_rank) {
        return emitOpError("output rank ") << output_rt.getRank()
               << " != params rank " << params_rank
               << " + indices rank " << indices_rank << " - 1 = "
               << expected_rank;
    }

    auto params_shape = params_rt.getShape();
    auto indices_shape = indices_rt.getShape();
    auto output_shape = output_rt.getShape();

    // verify shape: before axis, after axis
    for (i64 i = 0; i < axis; ++i) {
        if (params_shape[i] != output_shape[i]) {
            return emitOpError("output dim ") << i << " = "
                   << output_shape[i] << " != params dim " << i << " = "
                   << params_shape[i];
        }
    }
    for (i64 i = 0; i < indices_rank; ++i) {
        if (indices_shape[i] != output_shape[axis + i]) {
            return emitOpError("output dim ") << (axis + i) << " = "
                   << output_shape[axis + i]
                   << " != indices dim " << i << " = "
                   << indices_shape[i];
        }
    }
    for (i64 i = axis + 1; i < params_rank; ++i) {
        auto out_idx = i + indices_rank - 1;
        if (params_shape[i] != output_shape[out_idx]) {
            return emitOpError("output dim ") << out_idx << " = "
                   << output_shape[out_idx]
                   << " != params dim " << i << " = "
                   << params_shape[i];
        }
    }

    return mlir::success();
}

}
