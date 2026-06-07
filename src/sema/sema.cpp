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
}

void sema::resolve_func_decl(func_decl* f) {
    type return_type = ts.get_void_type();
    if (f->get_return_type()) {
        return_type = resolve_type(f->get_return_type());
        f->set_resolved(return_type);
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

    ctx.regist_function(f->get_name(), std::move(fi));
}

void sema::resolve_func_block(func_decl* f) {
    const auto& fi = ctx.get_functions().at(f->get_name());
    for (auto node : f->get_body()->get_stmts()) {
        resolve_stmt(node);
    }
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
