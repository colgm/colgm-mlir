#include "ast/visitor.hpp"

namespace colgm_mlir {

bool visitor::visit_ast(ast*) {
    return true;
}

bool visitor::visit_decl(decl*) {
    return true;
}

bool visitor::visit_stmt(stmt*) {
    return true;
}

bool visitor::visit_expr(expr*) {
    return true;
}

bool visitor::visit_root(root* n) {
    for (auto i : n->get_funcs()) {
        if (!i) {
            continue;
        }
        i->accept(this);
    }
    return true;
}

bool visitor::visit_int_literal(int_literal*) {
    return true;
}

bool visitor::visit_float_literal(float_literal*) {
    return true;
}

bool visitor::visit_bool_literal(bool_literal*) {
    return true;
}

bool visitor::visit_tensor(tensor* n) {
    for (auto i : n->get_values()) {
        if (!i) {
            continue;
        }
        i->accept(this);
    }
    return true;
}

bool visitor::visit_identifier(identifier*) {
    return true;
}

bool visitor::visit_binary_expr(binary_expr* n) {
    if (n->get_lhs()) {
        n->get_lhs()->accept(this);
    }
    if (n->get_rhs()) {
        n->get_rhs()->accept(this);
    }
    return true;
}

bool visitor::visit_unary_expr(unary_expr* n) {
    if (n->get_operand()) {
        n->get_operand()->accept(this);
    }
    return true;
}

bool visitor::visit_call_expr(call_expr* n) {
    if (n->get_callee()) {
        n->get_callee()->accept(this);
    }
    for (auto arg : n->get_args()) {
        if (!arg) {
            continue;
        }
        arg->accept(this);
    }
    return true;
}

bool visitor::visit_index_access(index_access* n) {
    if (n->get_target()) {
        n->get_target()->accept(this);
    }
    if (n->get_index()) {
        n->get_index()->accept(this);
    }
    return true;
}

bool visitor::visit_range_expr(range_expr* n) {
    if (n->get_start()) {
        n->get_start()->accept(this);
    }
    if (n->get_end()) {
        n->get_end()->accept(this);
    }
    return true;
}

bool visitor::visit_var_decl(var_decl* n) {
    if (n->get_init()) {
        n->get_init()->accept(this);
    }
    return true;
}

bool visitor::visit_assign_stmt(assign_stmt* n) {
    if (n->get_lhs()) {
        n->get_lhs()->accept(this);
    }
    if (n->get_rhs()) {
        n->get_rhs()->accept(this);
    }
    return true;
}

bool visitor::visit_return_stmt(return_stmt* n) {
    if (n->get_value()) {
        n->get_value()->accept(this);
    }
    return true;
}

bool visitor::visit_if_stmt(if_stmt* n) {
    if (n->get_condition()) {
        n->get_condition()->accept(this);
    }
    if (n->get_body()) {
        n->get_body()->accept(this);
    }
    if (n->get_else_body()) {
        n->get_else_body()->accept(this);
    }
    return true;
}

bool visitor::visit_for_stmt(for_stmt* n) {
    if (n->get_range()) {
        n->get_range()->accept(this);
    }
    if (n->get_body()) {
        n->get_body()->accept(this);
    }
    return true;
}

bool visitor::visit_block_stmt(block_stmt* n) {
    for (auto i : n->get_stmts()) {
        if (!i) {
            continue;
        }
        i->accept(this);
    }
    return true;
}

bool visitor::visit_func_decl(func_decl* n) {
    for (auto i : n->get_params()) {
        if (!i) {
            continue;
        }
        i->accept(this);
    }
    if (n->get_return_type()) {
        n->get_return_type()->accept(this);
    }
    if (n->get_body()) {
        n->get_body()->accept(this);
    }
    return true;
}

bool visitor::visit_type_def(type_def*) {
    return true;
}

bool visitor::visit_param(param* n) {
    if (n->get_type()) {
        n->get_type()->accept(this);
    }
    return true;
}

}
