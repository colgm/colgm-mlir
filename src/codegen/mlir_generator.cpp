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

void mlir_generator::generate_func(func_decl* f) {
    builder.setInsertionPointToStart(module.getBody());
    auto func = mlir::func::FuncOp::create(
        builder, to_loc(f), f->get_name(),
        builder.getFunctionType({}, {}));

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    generate_block(entry, f->get_body());

    builder.setInsertionPointToEnd(entry);
    mlir::func::ReturnOp::create(builder, to_loc(f));
}

void mlir_generator::generate_block(mlir::Block* entry, block_stmt* b) {
    for (auto i : b->get_stmts()) {
        generate_stmt(entry, i);
    }
}

void mlir_generator::generate_var_decl(mlir::Block* entry, var_decl* v) {
    generate_expr(entry, v->get_init());
}

void mlir_generator::generate_stmt(mlir::Block* entry, stmt* s) {
    if (s->is(ast_type::block_stmt)) {
        generate_block(entry, static_cast<block_stmt*>(s));
    } else if (s->is(ast_type::var_decl)) {
        generate_var_decl(entry, static_cast<var_decl*>(s));
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

mlir::Value mlir_generator::generate_expr(mlir::Block* entry, expr* e) {
    if (e->is(ast_type::int_literal)) {
        return generate_int_literal(static_cast<int_literal*>(e));
    } else if (e->is(ast_type::float_literal)) {
        return generate_float_literal(static_cast<float_literal*>(e));
    } else if (e->is(ast_type::bool_literal)) {
        return generate_bool_literal(static_cast<bool_literal*>(e));
    }

    assert(false && "not implemented");
    return mlir::Value();
}

void mlir_generator::generate(root* r) {
    module = mlir::ModuleOp::create(builder, to_loc(r));

    for (auto f : r->get_funcs()) {
        generate_func(f);
    }
}

}
