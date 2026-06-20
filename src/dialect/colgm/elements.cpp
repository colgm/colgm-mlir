#include <mlir/IR/OpImplementation.h>

#include "utils/type.hpp"
#include "dialect/colgm/elements.hpp"

namespace colgm_mlir {

void elements_op::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        mlir::ValueRange values, mlir::ArrayRef<int64_t> shape) {
    state.addOperands(values);

    auto operand_type = values[0].getType();
    auto op_rtt = llvm::cast<mlir::RankedTensorType>(operand_type);
    auto elt_type = op_rtt.getElementType();
    auto res_type = mlir::RankedTensorType::get(shape, elt_type);
    state.addTypes(res_type);

    state.addAttribute("target_shape", builder.getI64ArrayAttr(shape));
}

elements_op elements_op::create(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::ValueRange values, mlir::ArrayRef<int64_t> shape) {
    mlir::OperationState state(loc, getOperationName());
    build(builder, state, values, shape);
    return llvm::cast<elements_op>(builder.create(state));
}

mlir::ParseResult elements_op::parse(mlir::OpAsmParser& parser,
                                      mlir::OperationState& result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    mlir::Type elt_type, result_type;

    // (%v1, %v2, ...) {target_shape = [...]} : elt_type -> result_type
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(elt_type) ||
        parser.parseArrow() ||
        parser.parseType(result_type))
        return mlir::failure();

    for (auto& op : operands) {
        if (parser.resolveOperand(op, elt_type, result.operands))
            return mlir::failure();
    }

    result.addTypes(result_type);
    return mlir::success();
}

void elements_op::print(mlir::OpAsmPrinter& p) {
    p << "(";
    llvm::interleaveComma((*this)->getOperands(), p,
        [&](mlir::Value v) { p << v; });
    p << ") {target_shape = " << get_target_shape() << "}"
      << " : " << (*this)->getOperand(0).getType()
      << " -> " << (*this)->getResult(0).getType();
}

mlir::LogicalResult elements_op::verify() {
    auto num_operands = (*this)->getNumOperands();
    if (num_operands == 0) {
        return emitOpError("requires at least one operand");
    }

    auto operand_type = (*this)->getOperand(0).getType();
    auto op_rtt = llvm::dyn_cast<mlir::RankedTensorType>(operand_type);
    if (!op_rtt || op_rtt.getRank() != 0) {
        return emitOpError("operands must be rank-0 tensors, got ")
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

    auto output_shape = output_rt.getShape();
    if (target.size() != output_shape.size()) {
        return emitOpError("target_shape size ")
               << target.size() << " != output rank " << output_shape.size();
    }

    i64 total_elems = 1;
    for (u64 i = 0; i < target.size(); ++i) {
        auto dim = llvm::cast<mlir::IntegerAttr>(target[i]).getInt();
        if (dim != output_shape[i]) {
            return emitOpError("target_shape[") << i << "] = " << dim
                   << " != output dim " << i << " = " << output_shape[i];
        }
        total_elems *= dim;
    }

    if (static_cast<i64>(num_operands) != total_elems) {
        return emitOpError("operand count ") << num_operands
               << " != target_shape element count " << total_elems;
    }

    return mlir::success();
}

}
