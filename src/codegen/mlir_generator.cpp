#include <cassert>

#include "codegen/mlir_generator.hpp"

namespace colgm_mlir {

mlir::Type mlir_generator::convert_type(const type& t) {
    if (type::isa<void_type>(t) || type::isa<unknown_type>(t)) {
        return mlir::Type();
    }

    if (!type::isa<tensor_type>(t)) {
        assert(false && "not implemented");
        return {};
    }

    auto tt = type::as<tensor_type>(t);
    auto& shape = tt.get_shape();

    mlir::Type et;
    switch (tt.get_element_type().get_kind()) {
        case colgm_type_kind::TYPE_KIND_INT32: et = mlir::IntegerType::get(&ctx, 32); break;
        case colgm_type_kind::TYPE_KIND_INT64: et = mlir::IntegerType::get(&ctx, 64); break;
        case colgm_type_kind::TYPE_KIND_FLOAT32: et = mlir::Float32Type::get(&ctx); break;
        case colgm_type_kind::TYPE_KIND_FLOAT64: et = mlir::Float64Type::get(&ctx); break;
        case colgm_type_kind::TYPE_KIND_BOOL: et = mlir::IntegerType::get(&ctx, 1); break;
        default: assert(false && "not implemented"); break;
    }

    return mlir::RankedTensorType::get(shape, et);
}

mlir::Value mlir_generator::convert_index_value(mlir::Value val,
                                                const span& loc,
                                                const type t) {
    auto op = cast_op::create(builder, to_loc(loc), val, convert_type(t));
    return op->getResult(0);
}

void mlir_generator::flatten_tensor(std::vector<expr*>& v, tensor* n) {
    for (auto i : n->get_values()) {
        if (i->is(ast_type::tensor)) {
            flatten_tensor(v, static_cast<tensor*>(i));
        } else {
            v.push_back(i);
        }
    }
}

void mlir_generator::generate_func(func_decl* f) {
    vars.add_scope();

    auto func_ty = type::as<function_type>(f->get_resolved());
    llvm::SmallVector<mlir::Type> args;
    for (auto i : func_ty.get_arguments()) {
        args.push_back(convert_type(i));
    }

    auto ret_ty = convert_type(func_ty.get_return_type());
    llvm::SmallVector<mlir::Type> result_types;
    if (ret_ty) {
        result_types.push_back(ret_ty);
    }

    builder.setInsertionPointToStart(module->getBody());
    auto func = mlir::func::FuncOp::create(
        builder, to_loc(f), f->get_name(),
        builder.getFunctionType(args, result_types));

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // load arguments value
    auto block_args = entry->getArguments();
    auto& params = f->get_params();
    for (usize i = 0; i < params.size(); i++) {
        vars.add_var(params[i]->get_name(), block_args[i]);
    }

    generate_block(entry, f->get_body());

    if (entry->empty() || !entry->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.setInsertionPointToEnd(entry);
        mlir::func::ReturnOp::create(builder, to_loc(f));
    }

    vars.remove_scope();
}

void mlir_generator::generate_block(mlir::Block* entry, block_stmt* b) {
    vars.add_scope();
    for (auto i : b->get_stmts()) {
        generate_stmt(entry, i);
    }
    vars.remove_scope();
}

void mlir_generator::generate_var_decl(mlir::Block* entry, var_decl* v) {
    std::vector<mlir::Value> values;
    generate_expr_tuple(v->get_init(), values);
    assert(values.size() == v->get_vars().size());
    for (usize i = 0; i < v->get_vars().size(); i++) {
        vars.add_var(v->get_vars()[i], values[i]);
    }
}

void mlir_generator::generate_expr_stmt(mlir::Block* entry, expr_stmt* s) {
    auto inner = s->get_inner();
    if (!inner) return;
    std::vector<mlir::Value> values;
    generate_expr_tuple(inner, values);
}

void mlir_generator::generate_return_stmt(mlir::Block* entry, return_stmt* r) {
    if (r->get_value()) {
        auto value = generate_expr(r->get_value());
        mlir::func::ReturnOp::create(builder, to_loc(r), value);
    } else {
        mlir::func::ReturnOp::create(builder, to_loc(r));
    }
}

void mlir_generator::generate_yield_stmt(mlir::Block* entry, yield_stmt* y) {
    llvm::SmallVector<mlir::Value> elems;
    for (auto i : y->get_values()) {
        elems.push_back(generate_expr(i));
    }
    yield_op::create(builder, to_loc(y), elems);
}

void mlir_generator::generate_stmt(mlir::Block* entry, stmt* s) {
    if (s->is(ast_type::block_stmt)) {
        generate_block(entry, static_cast<block_stmt*>(s));
    } else if (s->is(ast_type::var_decl)) {
        generate_var_decl(entry, static_cast<var_decl*>(s));
    } else if (s->is(ast_type::return_stmt)) {
        generate_return_stmt(entry, static_cast<return_stmt*>(s));
    } else if (s->is(ast_type::yield_stmt)) {
        generate_yield_stmt(entry, static_cast<yield_stmt*>(s));
    } else if (s->is(ast_type::expr_stmt)) {
        generate_expr_stmt(entry, static_cast<expr_stmt*>(s));
    } else {
        assert(false && "not implemented");
    }
}

mlir::Value mlir_generator::generate_int_literal(int_literal* n) {
    auto ty = convert_type(n->get_resolved());
    std::vector<i64> vals = { n->get_literal() };
    auto attr = mlir::DenseElementsAttr::get<i64>(
        llvm::cast<mlir::RankedTensorType>(ty), vals);
    auto op = constant_op::create(builder, to_loc(n), attr, ty);
    return op->getResult(0);
}

mlir::Value mlir_generator::generate_float_literal(float_literal* n) {
    auto ty = convert_type(n->get_resolved());
    std::vector<f64> vals = { n->get_literal() };
    auto attr = mlir::DenseElementsAttr::get<f64>(
        llvm::cast<mlir::RankedTensorType>(ty), vals);
    auto op = constant_op::create(builder, to_loc(n), attr, ty);
    return op->getResult(0);
}

mlir::Value mlir_generator::generate_bool_literal(bool_literal* n) {
    auto ty = convert_type(n->get_resolved());
    auto attr = mlir::DenseElementsAttr::get(
        llvm::cast<mlir::RankedTensorType>(ty),
        llvm::ArrayRef<i64>{ static_cast<i64>(n->get_flag() ? 1 : 0) });
    auto op = constant_op::create(builder, to_loc(n), attr, ty);
    return op->getResult(0);
}

mlir::Value mlir_generator::generate_tensor(tensor* t) {
    std::vector<expr*> values;
    flatten_tensor(values, t);

    llvm::SmallVector<mlir::Value> elems;
    for (auto i : values) {
        elems.push_back(generate_expr(i));
    }

    auto full_ty = convert_type(t->get_resolved());
    auto full_shape = llvm::cast<mlir::RankedTensorType>(full_ty).getShape();
    auto operand_shape = llvm::cast<mlir::RankedTensorType>(elems[0].getType()).getShape();

    if (operand_shape.empty()) {
        // rank-0 operands: target_shape = full_shape
        auto op = elements_op::create(builder, to_loc(t), elems, full_shape);
        return op->getResult(0);
    } else {
        // rank>0 operands: target_shape = outer nesting dims
        auto target_rank = full_shape.size() - operand_shape.size();
        llvm::SmallVector<i64> target_shape(
            full_shape.begin(), full_shape.begin() + target_rank);
        auto op = stack_op::create(builder, to_loc(t), elems, target_shape);
        return op->getResult(0);
    }
}

mlir::Value mlir_generator::generate_identifier(identifier* i) {
    return vars.get_var(i->get_name());
}

mlir::Value mlir_generator::generate_binary_expr(binary_expr* n) {
    auto lhs = generate_expr(n->get_lhs());
    auto rhs = generate_expr(n->get_rhs());

    // cast index type to normal type if iter arg is used in the binary expression
    if (llvm::isa<mlir::IndexType>(lhs.getType()) && !llvm::isa<mlir::IndexType>(rhs.getType())) {
        lhs = convert_index_value(
            lhs,
            n->get_lhs()->get_location(),
            n->get_lhs()->get_resolved()
        );
    }
    if (llvm::isa<mlir::IndexType>(rhs.getType()) && !llvm::isa<mlir::IndexType>(lhs.getType())) {
        rhs = convert_index_value(
            rhs,
            n->get_rhs()->get_location(),
            n->get_rhs()->get_resolved()
        );
    }

    switch (n->get_op_type()) {
        case binary_expr::op::add: {
            auto op = add_op::create(builder, to_loc(n), lhs, rhs);
            return op->getResult(0);
        }
        case binary_expr::op::sub: {
            auto op = sub_op::create(builder, to_loc(n), lhs, rhs);
            return op->getResult(0);
        }
        case binary_expr::op::mul: {
            auto op = mul_op::create(builder, to_loc(n), lhs, rhs);
            return op->getResult(0);
        }
        case binary_expr::op::div: {
            auto op = div_op::create(builder, to_loc(n), lhs, rhs);
            return op->getResult(0);
        }
        case binary_expr::op::cmp_eq: {
            auto op = cmp_eq_op::create(builder, to_loc(n), lhs, rhs,
                                        convert_type(n->get_resolved()));
            return op->getResult(0);
        }
        case binary_expr::op::cmp_ne: {
            auto op = cmp_ne_op::create(builder, to_loc(n), lhs, rhs,
                                        convert_type(n->get_resolved()));
            return op->getResult(0);
        }
        case binary_expr::op::cmp_lt: {
            auto op = cmp_lt_op::create(builder, to_loc(n), lhs, rhs,
                                        convert_type(n->get_resolved()));
            return op->getResult(0);
        }
        case binary_expr::op::cmp_gt: {
            auto op = cmp_gt_op::create(builder, to_loc(n), lhs, rhs,
                                        convert_type(n->get_resolved()));
            return op->getResult(0);
        }
        case binary_expr::op::cmp_le: {
            auto op = cmp_le_op::create(builder, to_loc(n), lhs, rhs,
                                        convert_type(n->get_resolved()));
            return op->getResult(0);
        }
        case binary_expr::op::cmp_ge: {
            auto op = cmp_ge_op::create(builder, to_loc(n), lhs, rhs,
                                        convert_type(n->get_resolved()));
            return op->getResult(0);
        }
    }
    return mlir::Value();
}

mlir::Value mlir_generator::generate_unary_expr(unary_expr* n) {
    auto v = generate_expr(n->get_operand());

    if (llvm::isa<mlir::IndexType>(v.getType())) {
        v = convert_index_value(
            v,
            n->get_operand()->get_location(),
            n->get_resolved()
        );
    }

    switch (n->get_op_type()) {
        case unary_expr::op::sub: {
            auto op = neg_op::create(builder, to_loc(n), v);
            return op->getResult(0);
        }
    }

    return mlir::Value();
}

mlir::Value mlir_generator::generate_call_expr(call_expr* n) {
    if (!n->get_callee()->is(ast_type::identifier)) {
        assert(false && "not implemented");
    }

    if (type::isa<unknown_type>(n->get_callee()->get_resolved())) {
        assert(false && "not implemented");
    }

    llvm::SmallVector<mlir::Value> args;
    for (auto i : n->get_args()) {
        auto arg = generate_expr(i);
        if (llvm::isa<mlir::IndexType>(arg.getType())) {
            arg = convert_index_value(
                arg,
                i->get_location(),
                i->get_resolved()
            );
        }
        args.push_back(arg);
    }

    auto f = static_cast<identifier*>(n->get_callee())->get_name();
    auto ret_type = convert_type(n->get_resolved());
    auto call = mlir::func::CallOp::create(builder, to_loc(n), f, { ret_type }, args);
    return call.getResult(0);
}

mlir::Value mlir_generator::generate_index_access(index_access* n) {
    auto target = generate_expr(n->get_target());
    auto index = generate_expr(n->get_index());
    auto op = slice_op::create(builder, to_loc(n), target, index, 0);
    return op->getResult(0);
}

void mlir_generator::generate_if_expr(if_expr* i, std::vector<mlir::Value>& values) {
    auto cond = generate_expr(i->get_condition());

    auto i1_ty = mlir::IntegerType::get(&ctx, 1);
    auto cast = cast_op::create(builder, to_loc(i), cond, i1_ty);
    auto cond_i1 = cast->getResult(0);

    auto resolved = i->get_resolved();
    if_op op;

    // create if op
    if (type::isa<tuple_type>(resolved)) {
        auto& tup = type::as<tuple_type>(resolved);
        llvm::SmallVector<mlir::Type> result_types;
        for (auto& t : tup.get_types()) {
            result_types.push_back(convert_type(t));
        }
        op = if_op::create(builder, to_loc(i), cond_i1, result_types);
    } else {
        auto result_type = convert_type(resolved);
        mlir::TypeRange types = result_type ? mlir::TypeRange(result_type) : mlir::TypeRange();
        op = if_op::create(builder, to_loc(i), cond_i1, types);
    }

    // then
    {
        auto& block = op.get_then_region().front();
        builder.setInsertionPointToEnd(&block);
        generate_block(&block, i->get_body());
        if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            yield_op::create(builder, to_loc(i));
        }
    }

    if (auto* else_body = i->get_else_body()) {
        auto& block = op.get_else_region().front();
        builder.setInsertionPointToEnd(&block);
        generate_block(&block, else_body);
        if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            yield_op::create(builder, to_loc(i));
        }
    }

    builder.setInsertionPointAfter(op);

    for (auto r : op.getResults()) {
        values.push_back(r);
    }
}

void mlir_generator::generate_for_expr(for_expr* f, std::vector<mlir::Value>& values) {
    llvm::SmallVector<mlir::Value, 4> init_values;

    vars.add_scope();
    for (auto i : f->get_init_pairs()) {
        auto v = generate_expr(std::get<1>(i));
        init_values.push_back(v);
    }

    auto start = generate_expr(f->get_range()->get_start());
    auto end = generate_expr(f->get_range()->get_end());

    auto idx_ty = mlir::IndexType::get(&ctx);
    auto start_idx = cast_op::create(builder, to_loc(f), start, idx_ty)->getResult(0);
    auto end_idx = cast_op::create(builder, to_loc(f), end, idx_ty)->getResult(0);

    for_op op;
    if (type::isa<tuple_type>(f->get_resolved())) {
        auto tup = type::as<tuple_type>(f->get_resolved());
        llvm::SmallVector<mlir::Type, 4> result_types;
        for (auto t : tup.get_types()) {
            result_types.push_back(convert_type(t));
        }
        op = for_op::create(builder, to_loc(f), start_idx, end_idx, init_values, result_types);
    } else {
        auto rt = convert_type(f->get_resolved());
        op = for_op::create(builder, to_loc(f), start_idx, end_idx, init_values,
                            rt ? mlir::TypeRange(rt) : mlir::TypeRange());
    }

    {
        auto& block = op.get_body().front();

        vars.add_var(f->get_iter(), block.getArgument(0));
        const auto& pairs = f->get_init_pairs();
        for (usize i = 0; i < pairs.size(); i++) {
            vars.add_var(std::get<0>(pairs[i]), block.getArgument(i + 1));
        }

        builder.setInsertionPointToEnd(&block);
        generate_block(&block, f->get_body());
        if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            yield_op::create(builder, to_loc(f));
        }
    }

    vars.remove_scope();

    builder.setInsertionPointAfter(op);
    for (auto r : op.getResults()) {
        values.push_back(r);
    }
    return;
}
mlir::Value mlir_generator::generate_expr(expr* e) {
    if (e->is(ast_type::int_literal)) {
        return generate_int_literal(static_cast<int_literal*>(e));
    } else if (e->is(ast_type::float_literal)) {
        return generate_float_literal(static_cast<float_literal*>(e));
    } else if (e->is(ast_type::bool_literal)) {
        return generate_bool_literal(static_cast<bool_literal*>(e));
    } else if (e->is(ast_type::tensor)) {
        return generate_tensor(static_cast<tensor*>(e));
    } else if (e->is(ast_type::identifier)) {
        return generate_identifier(static_cast<identifier*>(e));
    } else if (e->is(ast_type::binary_expr)) {
        return generate_binary_expr(static_cast<binary_expr*>(e));
    } else if (e->is(ast_type::unary_expr)) {
        return generate_unary_expr(static_cast<unary_expr*>(e));
    } else if (e->is(ast_type::call_expr)) {
        return generate_call_expr(static_cast<call_expr*>(e));
    } else if (e->is(ast_type::index_access)) {
        return generate_index_access(static_cast<index_access*>(e));
    }

    assert(false && "not implemented");
    return mlir::Value();
}

void mlir_generator::generate_expr_tuple(expr* e, std::vector<mlir::Value>& values) {
    if (e->is(ast_type::if_expr)) {
        generate_if_expr(static_cast<if_expr*>(e), values);
    } else if (e->is(ast_type::for_expr)) {
        generate_for_expr(static_cast<for_expr*>(e), values);
    } else {
        values.push_back(generate_expr(e));
    }
}

void mlir_generator::generate(root* r) {
    module = mlir::ModuleOp::create(builder, to_loc(r));

    for (auto f : r->get_funcs()) {
        generate_func(f);
    }
}

}
