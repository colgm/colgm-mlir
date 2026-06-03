#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/transpose.hpp"

namespace colgm_mlir {

void transpose_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                         mlir::Value input, mlir::ArrayRef<int64_t> permutation) {
    state.addOperands({input});

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto input_shape = input_type.getShape();

    llvm::SmallVector<int64_t> output_shape;
    for (auto p : permutation) {
        output_shape.push_back(input_shape[p]);
    }

    auto res_type = mlir::RankedTensorType::get(output_shape,
                                                input_type.getElementType());
    state.addTypes(res_type);
    state.addAttribute("permutation", builder.getI64ArrayAttr(permutation));
}

mlir::ParseResult transpose_op::parse(mlir::OpAsmParser& parser,
                                      mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::TensorType input_type, result_type;

    // %input {permutation = [...]} : input_type -> result_type
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

void transpose_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " {permutation = " << get_permutation() << "}"
      << " : " << get_input().getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult transpose_op::verify() {
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

    if (!(*this)->hasAttr("permutation")) {
        return emitOpError("missing 'permutation' attribute");
    }

    auto target = llvm::cast<mlir::ArrayAttr>((*this)->getAttr("permutation"));
    for (u64 i = 0; i < target.size(); ++i) {
        if (!llvm::isa<mlir::IntegerAttr>(target[i])) {
            return emitOpError("permutation[") << i << "] must be an integer";
        }
    }

    auto rank = input_rt.getRank();
    if (target.size() != static_cast<size_t>(rank)) {
        return emitOpError("permutation size ")
               << target.size() << " != input rank " << rank;
    }

    for (u64 i = 0; i < target.size(); ++i) {
        auto dim = llvm::cast<mlir::IntegerAttr>(target[i]).getInt();
        if (dim < 0 || dim >= rank) {
            return emitOpError("permutation[") << i << "] = " << dim
                   << " is out of range [0, " << rank << ")";
        }
    }

    // check dim duplication
    llvm::SmallBitVector seen(rank);
    for (u64 i = 0; i < target.size(); ++i) {
        auto dim = llvm::cast<mlir::IntegerAttr>(target[i]).getInt();
        if (seen.test(dim)) {
            return emitOpError("permutation[") << i << "] = " << dim
                   << " is duplicated";
        }
        seen.set(dim);
    }

    // verify output shape
    for (u64 i = 0; i < target.size(); ++i) {
        auto dim = llvm::cast<mlir::IntegerAttr>(target[i]).getInt();
        if (input_rt.getDimSize(dim) != output_rt.getDimSize(i)) {
            return emitOpError("permutation[") << i << "] = " << dim
                   << " maps input dim " << input_rt.getDimSize(dim)
                   << " to output dim " << i
                   << ", but output dim is " << output_rt.getDimSize(i);
        }
    }

    return mlir::success();
}

}
