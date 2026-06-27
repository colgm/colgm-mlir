#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/stack.hpp"

namespace colgm_mlir {

void stack_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                     mlir::ValueRange values, mlir::ArrayRef<i64> shape) {
    state.addOperands(values);

    auto operand_type = values[0].getType();
    auto op_rtt = llvm::cast<mlir::RankedTensorType>(operand_type);
    auto elt_type = op_rtt.getElementType();

    // result shape = target_shape + operand_shape
    llvm::SmallVector<i64> res_shape(shape.begin(), shape.end());
    res_shape.append(op_rtt.getShape().begin(), op_rtt.getShape().end());
    auto res_type = mlir::RankedTensorType::get(res_shape, elt_type);
    state.addTypes(res_type);

    state.addAttribute("target_shape", builder.getI64ArrayAttr(shape));
}

stack_op stack_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::ValueRange values, mlir::ArrayRef<i64> shape) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, values, shape);
    return llvm::cast<stack_op>(builder.create(state));
}

mlir::ParseResult stack_op::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    mlir::Type operand_type, result_type;

    // (%v1, %v2, ...) {target_shape = [...]} : operand_type -> result_type
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(operand_type) ||
        parser.parseArrow() ||
        parser.parseType(result_type))
        return mlir::failure();

    for (auto& op : operands) {
        if (parser.resolveOperand(op, operand_type, result.operands))
            return mlir::failure();
    }

    result.addTypes(result_type);
    return mlir::success();
}

void stack_op::print(mlir::OpAsmPrinter& p) {
    p << "(";
    llvm::interleaveComma((*this)->getOperands(), p,
        [&](mlir::Value v) { p << v; });
    p << ") {target_shape = " << get_target_shape() << "}"
      << " : " << (*this)->getOperand(0).getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult stack_op::verify() {
    auto num_operands = (*this)->getNumOperands();
    if (num_operands == 0) {
        return emitOpError("requires at least one operand");
    }

    auto operand_type = (*this)->getOperand(0).getType();
    auto op_rtt = llvm::dyn_cast<mlir::RankedTensorType>(operand_type);
    if (!op_rtt) {
        return emitOpError("operands must be ranked tensors, got ")
               << operand_type;
    }
    auto elt_type = op_rtt.getElementType();

    for (unsigned i = 1; i < num_operands; ++i) {
        if ((*this)->getOperand(i).getType() != operand_type) {
            return emitOpError("operand ") << i << " type "
                   << (*this)->getOperand(i).getType()
                   << " != operand 0 type " << operand_type;
        }
    }

    auto output_type = (*this)->getResult(0).getType();
    if (!llvm::isa<mlir::RankedTensorType>(output_type)) {
        return emitOpError("result must be a ranked tensor");
    }
    auto output_rt = llvm::cast<mlir::RankedTensorType>(output_type);

    if (elt_type != output_rt.getElementType()) {
        return emitOpError("element type ") << elt_type
               << " != result element type " << output_rt.getElementType();
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

    // result_shape = target_shape + operand_shape
    auto operand_shape = op_rtt.getShape();
    auto output_shape = output_rt.getShape();
    u64 expected_rank = target.size() + operand_shape.size();
    if (output_shape.size() != expected_rank) {
        return emitOpError("result rank ") << output_shape.size()
               << " != target_shape rank " << target.size()
               << " + operand rank " << operand_shape.size();
    }

    for (u64 i = 0; i < target.size(); ++i) {
        auto dim = llvm::cast<mlir::IntegerAttr>(target[i]).getInt();
        if (dim != output_shape[i]) {
            return emitOpError("target_shape[") << i << "] = " << dim
                   << " != result dim[" << i << "] = " << output_shape[i];
        }
    }

    for (u64 i = 0; i < operand_shape.size(); ++i) {
        if (operand_shape[i] != output_shape[target.size() + i]) {
            return emitOpError("operand dim[") << i << "] = " << operand_shape[i]
                   << " != result dim[" << (target.size() + i)
                   << "] = " << output_shape[target.size() + i];
        }
    }

    i64 target_elems = 1;
    for (u64 i = 0; i < target.size(); ++i) {
        target_elems *= llvm::cast<mlir::IntegerAttr>(target[i]).getInt();
    }

    if (static_cast<i64>(num_operands) != target_elems) {
        return emitOpError("operand count ") << num_operands
               << " != product of target_shape dims " << target_elems;
    }

    return mlir::success();
}

}
