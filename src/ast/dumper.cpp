#include "ast/dumper.hpp"
#include "utils/colorful.hpp"

namespace colgm_mlir {

bool dumper::visit_root(root* node) {
    dump_indent();
    std::cout << purple << "Root" << reset << format_info(node);
    push_indent();
    for (auto i : node->get_funcs()) {
        if (i == node->get_funcs().back()) {
            set_last();
        }
        i->accept(this);
    }
    pop_indent();
    return true;
}

bool dumper::visit_int_literal(int_literal* node) {
    dump_indent();
    std::cout << purple << "IntLiteral " << reset;
    std::cout << green << node->get_literal() << reset;
    std::cout << format_info(node);
    return true;
}

bool dumper::visit_float_literal(float_literal* node) {
    dump_indent();
    std::cout << purple << "FloatLiteral " << reset;
    std::cout << green << node->get_literal() << reset << format_info(node);
    return true;
}

bool dumper::visit_bool_literal(bool_literal* node) {
    dump_indent();
    std::cout << purple << "BoolLiteral " << reset;
    std::cout << green << (node->get_flag() ? "true" : "false") << reset;
    std::cout << format_info(node);
    return true;
}

bool dumper::visit_tensor(tensor* node) {
    dump_indent();
    std::cout << purple << "Tensor" << reset << format_info(node);
    push_indent();
    for (auto i : node->get_values()) {
        if (i == node->get_values().back()) {
            set_last();
        }
        i->accept(this);
    }
    pop_indent();
    return true;
}

bool dumper::visit_identifier(identifier* node) {
    dump_indent();
    std::cout << purple << "identifier " << reset;
    std::cout << green << node->get_name() << reset << format_info(node);
    return true;
}

bool dumper::visit_binary_expr(binary_expr* node) {
    dump_indent();
    std::cout << purple << "BinaryExpr " << reset;
    switch (node->get_op_type()) {
        case binary_expr::op::add: std::cout << "+"; break;
        case binary_expr::op::sub: std::cout << "-"; break;
        case binary_expr::op::mul: std::cout << "*"; break;
        case binary_expr::op::div: std::cout << "/"; break;
    }
    std::cout << format_info(node);
    push_indent();
    node->get_lhs()->accept(this);
    set_last();
    node->get_rhs()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_unary_expr(unary_expr* node) {
    dump_indent();
    std::cout << purple << "UnaryExpr " << reset;
    switch (node->get_op_type()) {
        case unary_expr::op::sub: std::cout << "-"; break;
    }
    std::cout << format_info(node);
    push_indent();
    set_last();
    node->get_operand()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_call_expr(call_expr* node) {
    dump_indent();
    std::cout << purple << "CallExpr" << reset << format_info(node);
    push_indent();
    node->get_callee()->accept(this);
    for (auto i : node->get_args()) {
        if (i == node->get_args().back()) {
            set_last();
        }
        i->accept(this);
    }
    pop_indent();
    return true;
}

bool dumper::visit_index_access(index_access* node) {
    dump_indent();
    std::cout << purple << "IndexAccess" << reset << format_info(node);
    push_indent();
    node->get_target()->accept(this);
    set_last();
    node->get_index()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_range_expr(range_expr* node) {
    dump_indent();
    std::cout << purple << "RangeExpr" << reset << format_info(node);
    push_indent();
    node->get_start()->accept(this);
    set_last();
    node->get_end()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_var_decl(var_decl* node) {
    dump_indent();
    std::cout << purple << "VarDecl " << reset;
    std::cout << green << node->get_name() << reset << format_info(node);
    push_indent();
    set_last();
    node->get_init()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_return_stmt(return_stmt* node) {
    dump_indent();
    std::cout << purple << "ReturnStmt" << reset << format_info(node);
    push_indent();
    if (node->get_value()) {
        set_last();
        node->get_value()->accept(this);
    }
    pop_indent();
    return true;
}

bool dumper::visit_if_stmt(if_stmt* node) {
    dump_indent();
    std::cout << purple << "IfStmt" << reset << format_info(node);
    push_indent();
    node->get_condition()->accept(this);
    if (!node->get_else_body()) {
        set_last();
    }
    node->get_body()->accept(this);
    if (node->get_else_body()) {
        set_last();
        dump_indent();
        std::cout << purple << "ElseStmt" << reset;
        std::cout << format_info(node->get_else_body());
        push_indent();
        set_last();
        node->get_else_body()->accept(this);
        pop_indent();
    }
    pop_indent();
    return true;
}

bool dumper::visit_for_stmt(for_stmt* node) {
    dump_indent();
    std::cout << purple << "ForStmt " << reset;
    std::cout << node->get_iter() << format_info(node);
    push_indent();
    node->get_range()->accept(this);
    set_last();
    node->get_body()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_block_stmt(block_stmt* node) {
    dump_indent();
    std::cout << purple << "BlockStmt" << reset << format_info(node);
    push_indent();
    for (auto i : node->get_stmts()) {
        if (i == node->get_stmts().back()) {
            set_last();
        }
        i->accept(this);
    }
    pop_indent();
    return true;
}

bool dumper::visit_func_decl(func_decl* node) {
    dump_indent();
    std::cout << purple << "FuncDecl " << reset;
    std::cout << node->get_name() << format_info(node);
    push_indent();
    if (!node->get_params().empty()) {
        dump_indent();
        std::cout << purple << "Params:\n" << reset;
        push_indent();
        for (auto i : node->get_params()) {
            if (i == node->get_params().back()) {
                set_last();
            }
            i->accept(this);
        }
        pop_indent();
    }
    if (node->get_return_type()) {
        dump_indent();
        std::cout << purple << "ReturnType:\n" << reset;
        push_indent();
        set_last();
        node->get_return_type()->accept(this);
        pop_indent();
    }
    set_last();
    node->get_body()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_type_def(type_def* node) {
    dump_indent();
    std::cout << purple << "TypeDef " << reset;
    std::cout << green << node->get_base();
    if (!node->get_dims().empty()) {
        std::cout << "[";
        i64 count = 0;
        for (auto i : node->get_dims()) {
            if (count == node->get_dims().size() - 1) {
                std::cout << i;
            } else {
                std::cout << i << ", ";
            }
        }
        std::cout << "]";
    }
    std::cout << reset << format_info(node);
    return true;
}

bool dumper::visit_param(param* node) {
    dump_indent();
    std::cout << purple << "Param " << reset;
    std::cout << node->get_name() << format_info(node);
    if (node->get_type()) {
        push_indent();
        set_last();
        node->get_type()->accept(this);
        pop_indent();
    }
    return true;
}

}
