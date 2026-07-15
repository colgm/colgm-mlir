#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/broadcast.hpp"

namespace colgm_mlir {

void broadcast_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                         mlir::Value input, mlir::ArrayRef<i64> shape) {
    state.addOperands({input});

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto elem_type = input_type.getElementType();
    auto res_type = mlir::RankedTensorType::get(shape, elem_type);
    state.addTypes(res_type);

    state.addAttribute("target_shape", builder.getI64ArrayAttr(shape));
}

broadcast_op broadcast_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value input, mlir::ArrayRef<i64> shape) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, input, shape);
    return llvm::cast<broadcast_op>(builder.create(state));
}

mlir::ParseResult broadcast_op::parse(mlir::OpAsmParser& parser,
                                      mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::TensorType input_type, result_type;

    // %input {target_shape = [...]} : input_type -> result_type
    if (parser.parseOperand(input) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(input_type) ||
        parser.parseArrow() ||
        parser.parseType(result_type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, input_type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(result_type);
    return mlir::success();
}

void broadcast_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " {target_shape = " << get_target_shape() << "}"
      << " : " << get_input().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult broadcast_op::verify() {
    auto input_type = get_input().getType();
    if (!llvm::isa<mlir::RankedTensorType>(input_type)) {
        return emitOpError("input must be a ranked tensor");
    }
    auto output_type = (*this)->getResult(0).getType();
    if (!llvm::isa<mlir::RankedTensorType>(output_type)) {
        return emitOpError("output must be a ranked tensor");
    }

    auto input_rt = llvm::cast<mlir::RankedTensorType>(input_type);
    auto output_rt = llvm::cast<mlir::RankedTensorType>(output_type);

    if (input_rt.getElementType() != output_rt.getElementType()) {
        return emitOpError("element type mismatch: input=")
               << input_rt.getElementType() << ", output="
               << output_rt.getElementType();
    }

    if (!(*this)->hasAttr("target_shape")) {
        return emitOpError("missing 'target_shape' attribute");
    }

    auto target = llvm::cast<mlir::ArrayAttr>((*this)->getAttr("target_shape"));
    for (u64 i = 0; i < target.size(); ++i) {
        if (!llvm::isa<mlir::IntegerAttr>(target[i])) {
            return emitOpError("target_shape[") << i << "] must be an integer";
        }
    }

    auto output_shape = output_rt.getShape();
    if (target.size() != output_shape.size()) {
        return emitOpError("target_shape size ")
               << target.size() << " != output rank " << output_shape.size();
    }
    for (u64 i = 0; i < target.size(); ++i) {
        auto dim = llvm::cast<mlir::IntegerAttr>(target[i]).getInt();
        if (dim != output_shape[i]) {
            return emitOpError("target_shape[") << i << "] = " << dim
                   << " != output dim " << i << " = " << output_shape[i];
        }
    }

    auto input_rank = input_rt.getRank();
    auto output_rank = output_rt.getRank();
    if (input_rank > output_rank) {
        return emitOpError("input rank ")
               << input_rank << " > output rank " << output_rank;
    }

    for (u64 i = 0; i < input_rank; ++i) {
        auto input_dim = input_rt.getDimSize(input_rank - 1 - i);
        auto output_dim = output_rt.getDimSize(output_rank - 1 - i);
        if (input_dim == output_dim) {
            continue;
        }
        if (input_dim < output_dim && input_dim == 1) {
            continue;
        }
        return emitOpError("input dim ")
               << input_dim << " != output dim " << output_dim;
    }

    return mlir::success();
}

}
