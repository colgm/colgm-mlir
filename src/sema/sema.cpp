#include "sema/sema.hpp"

namespace colgm_mlir {

type sema::resolve_type(type_def* t) {
    type base = ts.get_void_type();
    if (t->get_base() == "i32") {
        base = ts.get_i32_type();
    } else if (t->get_base() == "i64") {
        base = ts.get_i64_type();
    } else if (t->get_base() == "f32") {
        base = ts.get_f32_type();
    } else if (t->get_base() == "f64") {
        base = ts.get_f64_type();
    } else if (t->get_base() == "bool") {
        base = ts.get_bool_type();
    } else if (t->get_base() == "void") {
        base = ts.get_void_type();
    } else {
        base = ts.get_unknown_type();
        err.err(t->get_location(), "unknown type: " + t->get_base());
    }

    if (t->get_dims().empty()) {
        return base;
    }
    return ts.get_tensor_type(base, t->get_dims());
}

void sema::resolve_stmt(stmt* node) {
    switch (node->get_ast_type()) {
        case ast_type::var_decl:
            resolve_var_decl(static_cast<var_decl*>(node));
            break;
        case ast_type::assign_stmt:
            resolve_assign_stmt(static_cast<assign_stmt*>(node));
            break;
        case ast_type::return_stmt:
            resolve_return_stmt(static_cast<return_stmt*>(node));
            break;
        case ast_type::if_stmt:
            resolve_if_stmt(static_cast<if_stmt*>(node));
            break;
        case ast_type::for_stmt:
            resolve_for_stmt(static_cast<for_stmt*>(node));
            break;
        case ast_type::block_stmt:
            resolve_block_stmt(static_cast<block_stmt*>(node));
            break;
        default:
            assert(false && "unsupported ast type");
            break;
    }
}

void sema::resolve_var_decl(var_decl* node) {
    auto ty = resolve_expr(node->get_init());
    node->set_resolved(ty);
    ctx.regist_variable(node->get_name(), ty);
}

void sema::resolve_assign_stmt(assign_stmt* node) {
    auto lt = resolve_expr(node->get_lhs());
    auto rt = resolve_expr(node->get_rhs());
}

void sema::resolve_return_stmt(return_stmt* node) {
    if (node->get_value()) {
        resolve_expr(node->get_value());
    }
}

void sema::resolve_if_stmt(if_stmt* node) {
    resolve_expr(node->get_condition());
}

void sema::resolve_for_stmt(for_stmt* node) {
    resolve_range_expr(node->get_range());
}

void sema::resolve_block_stmt(block_stmt* node) {
    ctx.new_scope();
    for (auto node: node->get_stmts()) {
        resolve_stmt(node);
    }
    ctx.pop_scope();
}

type sema::resolve_int_literal(int_literal* node) {
    auto it = ts.get_i64_type();
    node->set_resolved(it);
    return it;
}

type sema::resolve_float_literal(float_literal* node) {
    auto ft = ts.get_f64_type();
    node->set_resolved(ft);
    return ft;
}

type sema::resolve_bool_literal(bool_literal* node) {
    auto bt = ts.get_bool_type();
    node->set_resolved(bt);
    return bt;
}

type sema::resolve_tensor(tensor* node) {
    if (node->get_values().empty()) {
        err.err(node->get_location(), "empty tensor literal");
        return ts.get_unknown_type();
    }

    std::vector<type> values;
    for (auto i : node->get_values()) {
        values.push_back(resolve_expr(i));
    }
    auto target_type = values.front();
    for (auto i : values) {
        if (i != target_type) {
            err.err(node->get_location(), "different types in tensor literal");
            return ts.get_unknown_type();
        }
    }

    i64 dim = values.size();
    if (type::isa<tensor_type>(target_type)) {
        auto tt = type::as<tensor_type>(target_type);
        std::vector<i64> shape = tt.get_shape();
        shape.push_back(dim);
        
        auto res = ts.get_tensor_type(tt.get_element_type(), shape);
        node->set_resolved(res);
        return res;
    }

    auto res = ts.get_tensor_type(target_type, {dim});
    node->set_resolved(res);
    return res;
}

type sema::resolve_identifier(identifier* node) {
    return ts.get_unknown_type();
}

type sema::resolve_binary_expr(binary_expr* node) {
    auto lt = resolve_expr(node->get_lhs());
    auto rt = resolve_expr(node->get_rhs());
    // TODO
    return lt;
}

type sema::resolve_unary_expr(unary_expr* node) {
    auto t = resolve_expr(node->get_operand());
    node->set_resolved(t);
    // TODO
    return t;
}

type sema::resolve_call_expr(call_expr* node) {
    auto callee = resolve_expr(node->get_callee());
    for (const auto& i : node->get_args()) {
        resolve_expr(i.value);
    }
    return ts.get_unknown_type();
}

type sema::resolve_index_access(index_access* node) {
    auto lt = resolve_expr(node->get_target());
    auto rt = resolve_expr(node->get_index());
    return ts.get_unknown_type();
}

type sema::resolve_range_expr(range_expr* node) {
    auto lt = resolve_expr(node->get_start());
    auto rt = resolve_expr(node->get_end());

    return ts.get_unknown_type();
}

type sema::resolve_expr(expr* node) {
    switch (node->get_ast_type()) {
        case ast_type::int_literal:
            return resolve_int_literal(static_cast<int_literal*>(node));
        case ast_type::float_literal:
            return resolve_float_literal(static_cast<float_literal*>(node));
        case ast_type::bool_literal:
            return resolve_bool_literal(static_cast<bool_literal*>(node));
        case ast_type::tensor:
            return resolve_tensor(static_cast<tensor*>(node));
        case ast_type::identifier:
            return resolve_identifier(static_cast<identifier*>(node));
        case ast_type::binary_expr:
            return resolve_binary_expr(static_cast<binary_expr*>(node));
        case ast_type::unary_expr:
            return resolve_unary_expr(static_cast<unary_expr*>(node));
        case ast_type::call_expr:
            return resolve_call_expr(static_cast<call_expr*>(node));
        case ast_type::index_access:
            return resolve_index_access(static_cast<index_access*>(node));
        case ast_type::range_expr:
            return resolve_range_expr(static_cast<range_expr*>(node));
        default:
            assert(false && "unsupported ast type");
            break;
    }

    return ts.get_unknown_type();
}

void sema::resolve_func_decl(func_decl* f) {
    type return_type = ts.get_void_type();
    if (f->get_return_type()) {
        return_type = resolve_type(f->get_return_type());
        f->get_return_type()->set_resolved(return_type);
    }

    std::vector<type> params;
    for (auto param: f->get_params()) {
        type param_type = ts.get_unknown_type();
        if (param->get_type()) {
            param_type = resolve_type(param->get_type());
        }
        param->set_resolved(param_type);
        params.push_back(param_type);
    }

    func_info fi(f->get_location(), ts.get_function_type(params, return_type));
    for (u64 i = 0; i < params.size(); i++) {
        fi.add_arg(f->get_params()[i]->get_name(), params[i]);
    }

    f->set_resolved(fi.func_type);
    ctx.regist_function(f->get_name(), std::move(fi));
}

void sema::resolve_func_block(func_decl* f) {
    const auto& fi = ctx.get_functions().at(f->get_name());
    ctx.new_scope();
    for (auto node : f->get_body()->get_stmts()) {
        resolve_stmt(node);
    }
    ctx.pop_scope();
}

const error& sema::scan(root* ast_root) {
    for (auto node: ast_root->get_funcs()) {
        resolve_func_decl(node);
    }
    for (auto node: ast_root->get_funcs()) {
        if (!ctx.get_functions().count(node->get_name())) {
            continue;
        }
        const auto& fi = ctx.get_functions().at(node->get_name());
        if (fi.is_generic_function) {
            continue;
        }
        resolve_func_block(node);
    }
    return err;
}

void sema::dump() const {
    ctx.dump();
}

}
