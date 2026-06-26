#include "sema/sema.hpp"

namespace colgm_mlir {

bool sema::block_contains_return(block_stmt* node) {
    for (auto i : node->get_stmts()) {
        if (i->is(ast_type::return_stmt)) {
            return true;
        }
    }
    return false;
}

yield_stmt* sema::find_yield(block_stmt* node) {
    for (auto i : node->get_stmts()) {
        if (i->is(ast_type::yield_stmt)) {
            return static_cast<yield_stmt*>(i);
        }
    }
    return nullptr;
}

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

    if (type::isa<tensor_type>(base)) {
        auto elem = type::as<tensor_type>(base).get_element_type();
        return ts.get_tensor_type(elem, t->get_dims());
    }
    return ts.get_tensor_type(base, t->get_dims());
}

void sema::resolve_stmt(stmt* node, bool allow_return) {
    switch (node->get_ast_type()) {
        case ast_type::var_decl:
            resolve_var_decl(static_cast<var_decl*>(node));
            break;
        case ast_type::return_stmt:
            if (!allow_return) {
                err.err(node->get_location(), "return statement not allowed in this scope");
                return;
            }
            resolve_return_stmt(static_cast<return_stmt*>(node));
            break;
        case ast_type::yield_stmt:
            resolve_yield_stmt(static_cast<yield_stmt*>(node));
            break;
        case ast_type::expr_stmt:
            resolve_expr_stmt(static_cast<expr_stmt*>(node));
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

    if (type::isa<tuple_type>(ty)) {
        auto tup = type::as<tuple_type>(ty);
        if (node->get_vars().size() != tup.get_types().size()) {
            err.err(node->get_location(), "tuple type must have same size as variable list");
            return;
        }

        for (usize i = 0; i < node->get_vars().size(); i++) {
            const auto& name = node->get_vars()[i];
            ctx.regist_variable(name, tup.get_types()[i]);
            if (tup.get_types()[i] == ts.get_void_type()) {
                err.err(node->get_location(), "variable '" + name + "' must have type but get void type");
                continue;
            }
        }
        return;
    }

    if (node->get_vars().size() != 1) {
        err.err(node->get_location(), "expect tuple type to be assigned to multiple variables");
        return;
    }

    const auto& name = node->get_vars()[0];
    if (ty == ts.get_void_type()) {
        err.err(node->get_location(), "variable '" + name + "' must have type but get void type");
    }
    ctx.regist_variable(name, ty);
}

void sema::resolve_return_stmt(return_stmt* node) {
    auto t = ts.get_void_type();
    if (node->get_value()) {
        t = resolve_expr(node->get_value());
    }
    node->set_resolved(t);
    if (t != func_ret_type) {
        err.err(
            node->get_location(),
            "return type must be '" + func_ret_type.to_string() + "' but get '" +
            t.to_string() + "'"
        );
    }
}

void sema::resolve_yield_stmt(yield_stmt* node) {
    if (node->get_values().empty()) {
        err.err(node->get_location(), "yield statement must have value");
        return;
    }

    std::vector<type> values;
    for (auto i : node->get_values()) {
        auto t = resolve_expr(i);
        values.push_back(t);
    }

    auto res = ts.get_tuple_type(values);
    node->set_resolved(res);
}

void sema::resolve_expr_stmt(expr_stmt* node) {
    auto e = node->get_inner();
    if (!e) return;
    resolve_expr(e);
    if (e->is(ast_type::if_expr) || e->is(ast_type::for_expr)) {
        if (e->get_resolved() != ts.get_void_type()) {
            err.warn(e->get_location(), "yield value ignored");
        }
    }
    node->set_resolved(e->get_resolved());
}

void sema::resolve_block_stmt(block_stmt* node) {
    ctx.new_scope();
    for (auto node: node->get_stmts()) {
        resolve_stmt(node, false);
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
    auto first_type = values.front();
    for (auto i : values) {
        if (i != first_type) {
            err.err(node->get_location(), "different types in tensor literal");
            return ts.get_unknown_type();
        }
    }

    std::vector<i64> shape = { static_cast<i64>(values.size()) };
    auto element_type = first_type;
    if (type::isa<tensor_type>(first_type)) {
        auto tt = type::as<tensor_type>(first_type);
        shape.insert(shape.end(), tt.get_shape().begin(), tt.get_shape().end());
        element_type = tt.get_element_type();
    }

    auto res = ts.get_tensor_type(element_type, shape);
    node->set_resolved(res);
    return res;
}

type sema::resolve_identifier(identifier* node) {
    auto res = ctx.find_variable(node->get_name());
    if (!res.found) {
        err.err(node->get_location(), "variable not found");
        return ts.get_unknown_type();
    }
    node->set_resolved(res.ty);
    return res.ty;
}

type sema::resolve_binary_expr(binary_expr* node) {
    auto lt = resolve_expr(node->get_lhs());
    auto rt = resolve_expr(node->get_rhs());
    if (lt != rt) {
        err.err(
            node->get_location(),
            "different types in binary expression, lhs: '" +
            lt.to_string() + "', rhs: '" + rt.to_string() + "'"
        );
    }

    auto res = lt;
    switch (node->get_op_type()) {
        case binary_expr::op::cmp_eq:
        case binary_expr::op::cmp_ne:
        case binary_expr::op::cmp_lt:
        case binary_expr::op::cmp_le:
        case binary_expr::op::cmp_gt:
        case binary_expr::op::cmp_ge: {
            res = ts.get_bool_type();
            if (type::isa<tensor_type>(lt)) {
                auto bool_base = type::as<tensor_type>(ts.get_bool_type()).get_element_type();
                auto ltt = type::as<tensor_type>(lt);
                res = ts.get_tensor_type(bool_base, ltt.get_shape());
            }
            break;
        }
        default: break;
    }
    node->set_resolved(res);
    return res;
}

type sema::resolve_unary_expr(unary_expr* node) {
    auto t = resolve_expr(node->get_operand());
    node->set_resolved(t);
    return t;
}

type sema::resolve_call_expr(call_expr* node) {
    if (node->get_callee()->is(ast_type::identifier)) {
        auto n = static_cast<identifier*>(node->get_callee());
        auto res = ctx.find_intrinsic(n->get_name());
        if (res.found) {
            for (auto i : node->get_args()) {
                resolve_expr(i);
            }
            n->set_resolved(ts.get_unknown_type());
            auto t = res.check(err, node, ts);
            node->set_resolved(t);
            return t;
        }
    }

    auto callee = resolve_expr(node->get_callee());
    if (!type::isa<function_type>(callee)) {
        err.err(node->get_location(), "not a function");
        return ts.get_unknown_type();
    }

    auto ft = type::as<function_type>(callee);
    if (ft.get_arguments().size() != node->get_args().size()) {
        err.err(node->get_location(), "different number of arguments");
        node->set_resolved(ft.get_return_type());
        return ft.get_return_type();
    }

    for (usize i = 0; i < ft.get_arguments().size(); ++i) {
        auto n = node->get_args()[i];
        auto nt = resolve_expr(n);
        auto expect = ft.get_arguments()[i];
        if (nt != expect) {
            err.err(n->get_location(), "different argument type in call expression");
        }
    }

    node->set_resolved(ft.get_return_type());
    return ft.get_return_type();
}

type sema::resolve_index_access(index_access* node) {
    auto lt = resolve_expr(node->get_target());
    auto rt = resolve_expr(node->get_index());

    if (!type::isa<tensor_type>(lt)) {
        err.err(node->get_location(), "not a tensor");
        return ts.get_unknown_type();
    }

    if (rt != ts.get_i32_type() && rt != ts.get_i64_type()) {
        err.err(node->get_index()->get_location(), "not a tensor<int> type");
        return ts.get_unknown_type();
    }

    auto tt = type::as<tensor_type>(lt);
    if (tt.get_shape().empty()) {
        err.err(node->get_location(), "cannot index a scalar");
        return ts.get_unknown_type();
    }

    std::vector<i64> shape(tt.get_shape().begin() + 1, tt.get_shape().end());
    auto res = ts.get_tensor_type(tt.get_element_type(), shape);
    node->set_resolved(res);
    return res;
}

type sema::resolve_range_expr(range_expr* node) {
    auto lt = resolve_expr(node->get_start());
    auto rt = resolve_expr(node->get_end());
    if (lt != rt) {
        err.err(
            node->get_location(),
            "different types in range, lhs '" +
            lt.to_string() + "', rhs '" +
            rt.to_string() + "'"
        );
    }
    if (lt != ts.get_i32_type() && lt != ts.get_i64_type()) {
        err.err(node->get_start()->get_location(), "not a tensor<int> type");
    }
    if (rt != ts.get_i32_type() && rt != ts.get_i64_type()) {
        err.err(node->get_end()->get_location(), "not a tensor<int> type");
    }

    node->set_resolved(lt);
    return lt;
}

type sema::resolve_if_expr(if_expr* node) {
    auto t = resolve_expr(node->get_condition());
    if (t != ts.get_bool_type()) {
        err.err(
            node->get_condition()->get_location(),
            "condition type must be bool"
        );
    }

    resolve_block_stmt(node->get_body());
    if (block_contains_return(node->get_body())) {
        err.err(node->get_location(),
            "cannot return in if expression",
            "maybe you mean 'yield'?"
        );
    }
    auto if_yield = find_yield(node->get_body());
    if (!if_yield) {
        node->set_resolved(ts.get_void_type());
        node->get_body()->set_resolved(ts.get_void_type());
    } else {
        node->set_resolved(if_yield->get_resolved());
        node->get_body()->set_resolved(if_yield->get_resolved());
    }

    if (if_yield && !node->get_else_body()) {
        err.err(node->get_location(),
            "all branches must yield same type but else branch does not exist"
        );
        return ts.get_unknown_type();
    }

    if (node->get_else_body()) {
        resolve_block_stmt(node->get_else_body());
        if (block_contains_return(node->get_else_body())) {
            err.err(node->get_location(),
                "cannot return in if expression",
                "maybe you mean 'yield'?"
            );
        }
        auto else_yield = find_yield(node->get_else_body());
        if (!else_yield) {
            node->set_resolved(ts.get_void_type());
        } else {
            node->get_else_body()->set_resolved(else_yield->get_resolved());
        }

        if (if_yield && else_yield &&
            else_yield->get_resolved() != if_yield->get_resolved()) {
            err.err(node->get_location(),
                "different yield types in if-else expression"
            );
            return ts.get_unknown_type();
        }
        if (if_yield && !else_yield) {
            err.err(node->get_location(),
                "all branches must yield same type but else branch yields void"
            );
            return ts.get_unknown_type();
        }
        if (!if_yield && else_yield) {
            err.err(node->get_location(),
                "all branches must yield same type but if branch yields void"
            );
            return ts.get_unknown_type();
        }
    }

    return node->get_resolved();
}

type sema::resolve_for_expr(for_expr* node) {
    auto rt = resolve_range_expr(node->get_range());
    if (rt != ts.get_i32_type() && rt != ts.get_i64_type()) {
        err.err(node->get_location(),
            "iterator type must be tensor<int> but get '" + rt.to_string() + "'"
        );
    }

    ctx.new_scope();
    ctx.regist_variable(node->get_iter(), rt);
    node->set_resolved(rt);

    for (const auto& i : node->get_init_pairs()) {
        auto t = resolve_expr(std::get<1>(i));
        ctx.regist_variable(std::get<0>(i), t);
    }

    resolve_block_stmt(node->get_body());
    if (block_contains_return(node->get_body())) {
        err.err(node->get_location(),
            "cannot return in for expression",
            "maybe you mean 'yield'?"
        );
    }

    auto yield_node = find_yield(node->get_body());
    if (!yield_node) {
        node->get_body()->set_resolved(ts.get_void_type());
    } else {
        node->get_body()->set_resolved(yield_node->get_resolved());
    }

    if (yield_node) {
        if (yield_node->get_values().size() != node->get_init_pairs().size()) {
            err.err(yield_node->get_location(),
                "yield values size does not match init pairs size"
            );
        } else {
            for (auto i = 0; i < yield_node->get_values().size(); i++) {
                auto expect = std::get<1>(node->get_init_pairs()[i])->get_resolved();
                auto actual = yield_node->get_values()[i]->get_resolved();

                if (expect != actual) {
                    err.err(yield_node->get_values()[i]->get_location(),
                        "yield value type does not match init pair type, "
                        "expect '" + expect.to_string() + "', "
                        "but got '" + actual.to_string() + "'"
                    );
                }
            }
        }
    } else if (node->get_init_pairs().size() > 0) {
        err.err(node->get_location(),
            "must yield same-size values because init pairs exist"
        );
    }

    if (node->get_init_pairs().empty()) {
        node->set_resolved(ts.get_void_type());
    } else {
        std::vector<type> types;
        for (const auto& i : node->get_init_pairs()) {
            types.push_back(std::get<1>(i)->get_resolved());
        }
        node->set_resolved(ts.get_tuple_type(types));
    }

    ctx.pop_scope();

    return node->get_resolved();
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
        case ast_type::if_expr:
            return resolve_if_expr(static_cast<if_expr*>(node));
        case ast_type::for_expr:
            return resolve_for_expr(static_cast<for_expr*>(node));
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
        } else {
            err.err(param->get_location(), "type not specified");
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
    func_ret_type = type::as<function_type>(fi.func_type).get_return_type();
    ctx.new_scope();

    // load parameters
    for (auto& name: fi.args) {
        ctx.regist_variable(name, fi.args_types.at(name));
    }

    for (auto node : f->get_body()->get_stmts()) {
        resolve_stmt(node, true);
    }

    if (auto n = find_yield(f->get_body())) {
        err.err(n->get_location(),
            "cannot use yield in function '" + f->get_name() + "'"
        );
    }

    if (!block_contains_return(f->get_body()) && func_ret_type != ts.get_void_type()) {
        err.err(f->get_location(),
            "function '" + f->get_name() + "' does not return value"
        );
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

        resolve_func_block(node);
    }
    return err;
}

void sema::dump() const {
    ctx.dump();
}

}
