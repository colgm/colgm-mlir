#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/reduce_sum.hpp"

namespace colgm_mlir {

void reduce_sum::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                       mlir::Value input, mlir::ArrayRef<i64> axes) {
    state.addOperands({input});

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto elem_type = input_type.getElementType();
    auto input_shape = input_type.getShape();

    llvm::SmallBitVector reduced(static_cast<unsigned>(input_shape.size()));
    for (auto axis : axes) {
        reduced.set(static_cast<unsigned>(axis));
    }

    llvm::SmallVector<i64> output_shape;
    for (u64 i = 0; i < input_shape.size(); ++i) {
        if (!reduced.test(i)) {
            output_shape.push_back(input_shape[i]);
        }
    }

    auto res_type = mlir::RankedTensorType::get(output_shape, elem_type);
    state.addTypes(res_type);
    state.addAttribute("axes", builder.getI64ArrayAttr(axes));
}

mlir::ParseResult reduce_sum::parse(mlir::OpAsmParser& parser,
                                    mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::TensorType input_type, result_type;

    // %input {axes = [...]} : input_type -> result_type
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

void reduce_sum::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " {axes = " << get_axes() << "}"
      << " : " << get_input().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult reduce_sum::verify() {
    auto input_type = get_input().getType();
    if (!llvm::isa<mlir::RankedTensorType>(input_type)) {
        return emitOpError("input must be a ranked tensor");
    }
    auto output_type = (*this)->getResult(0).getType();
    if (!llvm::isa<mlir::RankedTensorType>(output_type)) {
        return emitOpError("output must be a ranked tensor");
    }

    auto input_rt = llvm::cast<mlir::RankedTensorType>(input_type);
    auto input_shape = input_rt.getShape();
    auto output_rt = llvm::cast<mlir::RankedTensorType>(output_type);
    auto output_shape = output_rt.getShape();

    if (input_rt.getElementType() != output_rt.getElementType()) {
        return emitOpError("element type mismatch: input=")
               << input_rt.getElementType() << ", output="
               << output_rt.getElementType();
    }

    if (!(*this)->hasAttr("axes")) {
        return emitOpError("missing 'axes' attribute");
    }

    auto axes = llvm::cast<mlir::ArrayAttr>((*this)->getAttr("axes"));
    for (u64 i = 0; i < axes.size(); ++i) {
        if (!llvm::isa<mlir::IntegerAttr>(axes[i])) {
            return emitOpError("axes[") << i << "] must be an integer";
        }
        auto axis = llvm::cast<mlir::IntegerAttr>(axes[i]).getInt();
        if (axis < 0 || axis >= input_rt.getRank()) {
            return emitOpError("axes[") << i << "] = " << axis
                   << " is out of range [0, " << input_rt.getRank() << ")";
        }
    }

    llvm::SmallBitVector reduced(static_cast<unsigned>(input_rt.getRank()));
    for (auto axis : axes) {
        auto i = static_cast<unsigned>(llvm::cast<mlir::IntegerAttr>(axis).getInt());
        if (reduced.test(i)) {
            return emitOpError("axes[") << i << "] = " << axis
                   << " is duplicated";
        }
        reduced.set(i);
    }

    llvm::SmallVector<i64> expected_shape;
    for (u64 i = 0; i < input_shape.size(); ++i) {
        if (!reduced.test(i)) {
            expected_shape.push_back(input_shape[i]);
        }
    }

    if (expected_shape != output_shape) {
        return emitOpError("output shape mismatch: expected=[")
               << expected_shape << "], actual=[" << output_shape << "]";
    }

    return mlir::success();
}

}
