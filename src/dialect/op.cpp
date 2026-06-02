#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/op.hpp"

namespace colgm_mlir {

void add_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

mlir::ParseResult add_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void add_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void sub_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

mlir::ParseResult sub_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void sub_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void mul_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

mlir::ParseResult mul_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void mul_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void div_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType()); // SameOperandsAndResultType requires this
}

mlir::ParseResult div_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type type;

    // %lhs, %rhs : type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, type, result.operands) ||
        parser.resolveOperand(rhs, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void div_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType();
}

void matmul_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = llvm::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = llvm::cast<mlir::RankedTensorType>(rhs.getType());
    auto elem_type = lhs_type.getElementType();
    auto res_type = mlir::RankedTensorType::get({
        lhs_type.getDimSize(0),
        rhs_type.getDimSize(1),
    }, elem_type);
    state.addOperands({lhs, rhs});
    state.addTypes(res_type);
}

mlir::ParseResult matmul_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
    mlir::Type lhs_type, rhs_type;

    // %lhs, %rhs : type, type
    if (parser.parseOperand(lhs) || parser.parseComma() ||
        parser.parseOperand(rhs) || parser.parseColonType(lhs_type) ||
        parser.parseComma() || parser.parseType(rhs_type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(lhs, lhs_type, result.operands) ||
        parser.resolveOperand(rhs, rhs_type, result.operands)) {
        return mlir::failure();
    }

    auto elem_type = llvm::cast<mlir::RankedTensorType>(lhs_type).getElementType();
    auto res_type = mlir::RankedTensorType::get({
        llvm::cast<mlir::RankedTensorType>(lhs_type).getDimSize(0),
        llvm::cast<mlir::RankedTensorType>(rhs_type).getDimSize(1),
    }, elem_type);
    result.addTypes(res_type);
    return mlir::success();
}

void matmul_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_lhs() << ", " << get_rhs()
      << " : " << get_lhs().getType() << ", " << get_rhs().getType();
}

mlir::LogicalResult matmul_op::verify() {
    auto lhs_type = llvm::cast<mlir::RankedTensorType>(get_lhs().getType());
    auto rhs_type = llvm::cast<mlir::RankedTensorType>(get_rhs().getType());
    if (lhs_type.getRank() != 2 || rhs_type.getRank() != 2) {
        return emitOpError("only support 2D tensor");
    }
    if (lhs_type.getDimSize(1) != rhs_type.getDimSize(0)) {
        return emitOpError("lhs cols must equal rhs rows");
    }
    return mlir::success();
}

void relu_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult relu_op::parse(mlir::OpAsmParser& parser,
                                 mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    // %input : type
    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void relu_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void neg_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult neg_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void neg_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void abs_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult abs_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void abs_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void exp_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult exp_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void exp_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void log_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult log_op::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void log_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void sqrt_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult sqrt_op::parse(mlir::OpAsmParser& parser,
                                 mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void sqrt_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void sigmoid_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                       mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult sigmoid_op::parse(mlir::OpAsmParser& parser,
                                    mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void sigmoid_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

void tanh_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                    mlir::Value input) {
    state.addOperands({input});
    state.addTypes(input.getType());
}

mlir::ParseResult tanh_op::parse(mlir::OpAsmParser& parser,
                                 mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand input;
    mlir::Type type;

    if (parser.parseOperand(input) || parser.parseColonType(type)) {
        return mlir::failure();
    }

    if (parser.resolveOperand(input, type, result.operands)) {
        return mlir::failure();
    }

    result.addTypes(type);
    return mlir::success();
}

void tanh_op::print(mlir::OpAsmPrinter& p) {
    p << " " << get_input()
      << " : " << get_input().getType();
}

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

void broadcast_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                         mlir::Value input, mlir::ArrayRef<int64_t> shape) {
    state.addOperands({input});

    auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
    auto elem_type = input_type.getElementType();
    auto res_type = mlir::RankedTensorType::get(shape, elem_type);
    state.addTypes(res_type);

    state.addAttribute("target_shape", builder.getI64ArrayAttr(shape));
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