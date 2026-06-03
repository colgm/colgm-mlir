#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/reshape.hpp"

namespace colgm_mlir {

void reshape_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                       mlir::Value input, mlir::ArrayRef<int64_t> shape) {
    state.addOperands({input});

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto elem_type = input_type.getElementType();
    auto res_type = mlir::RankedTensorType::get(shape, elem_type);
    state.addTypes(res_type);

    state.addAttribute("target_shape", builder.getI64ArrayAttr(shape));
}

mlir::ParseResult reshape_op::parse(mlir::OpAsmParser& parser,
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

void reshape_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " {target_shape = " << get_target_shape() << "}"
      << " : " << get_input().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult reshape_op::verify() {
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
    auto output_shape = output_rt.getShape();
    if (target.size() != output_shape.size()) {
        return emitOpError("shape mismatch");
    }

    for (u64 i = 0; i < target.size(); ++i) {
        if (!llvm::isa<mlir::IntegerAttr>(target[i])) {
            return emitOpError("target_shape[") << i << "] must be an integer";
        }
        auto dim_attr = llvm::cast<mlir::IntegerAttr>(target[i]);
        if (dim_attr.getInt() != output_shape[i]) {
            return emitOpError("target_shape[") << i << "] = " << dim_attr.getInt()
                   << " != output[" << i << "] = " << output_shape[i];
        }
    }

    i64 input_elems = 1, output_elems = 1;
    for (auto dim : input_rt.getShape()) {
        input_elems *= dim;
    }
    for (auto dim : output_rt.getShape()) {
        output_elems *= dim;
    }
    if (input_elems != output_elems) {
        return emitOpError("total elements mismatch");
    }
    return mlir::success();
}

}
