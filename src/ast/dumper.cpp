#include "ast/dumper.hpp"

namespace colgm_mlir {

bool dumper::visit_root(root* node) {
    dump_indent();
    std::cout << "Root" << format_location(node);
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
    std::cout << "IntLiteral " << node->get_literal() << format_location(node);
    return true;
}

bool dumper::visit_float_literal(float_literal* node) {
    dump_indent();
    std::cout << "FloatLiteral " << node->get_literal() << format_location(node);
    return true;
}

bool dumper::visit_bool_literal(bool_literal* node) {
    dump_indent();
    std::cout << "BoolLiteral " << (node->get_flag()? "true":"false") << format_location(node);
    return true;
}

bool dumper::visit_identifier(identifier* node) {
    dump_indent();
    std::cout << "identifier " << node->get_name() << format_location(node);
    return true;
}

bool dumper::visit_binary_expr(binary_expr* node) {
    dump_indent();
    std::cout << "BinaryExpr ";
    switch (node->get_op_type()) {
        case binary_expr::op::add: std::cout << "+"; break;
        case binary_expr::op::sub: std::cout << "-"; break;
        case binary_expr::op::mul: std::cout << "*"; break;
        case binary_expr::op::div: std::cout << "/"; break;
    }
    std::cout << format_location(node);
    push_indent();
    node->get_lhs()->accept(this);
    set_last();
    node->get_rhs()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_unary_expr(unary_expr* node) {
    dump_indent();
    std::cout << "UnaryExpr ";
    switch (node->get_op_type()) {
        case unary_expr::op::add: std::cout << "+"; break;
        case unary_expr::op::sub: std::cout << "-"; break;
    }
    std::cout << format_location(node);
    push_indent();
    set_last();
    node->get_operand()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_call_expr(call_expr* node) {
    dump_indent();
    std::cout << "CallExpr " << format_location(node);
    push_indent();
    node->get_callee()->accept(this);
    for (auto i : node->get_args()) {
        if (i == node->get_args().back()) {
            set_last();
        }
        if (i.name.has_value()) {
            dump_indent();
            std::cout << "Arg " << i.name.value() << ": ";
            push_indent();
            set_last();
            i.value->accept(this);
            pop_indent();
        } else {
            i.value->accept(this);
        }
    }
    pop_indent();
    return true;
}

bool dumper::visit_index_access(index_access* node) {
    dump_indent();
    std::cout << "IndexAccess " << format_location(node);
    push_indent();
    node->get_target()->accept(this);
    set_last();
    node->get_index()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_range_expr(range_expr* node) {
    dump_indent();
    std::cout << "RangeExpr " << format_location(node);
    push_indent();
    node->get_start()->accept(this);
    set_last();
    node->get_end()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_var_decl(var_decl* node) {
    dump_indent();
    std::cout << "VarDecl " << node->get_name() << format_location(node);
    push_indent();
    set_last();
    node->get_init()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_assign_stmt(assign_stmt* node) {
    dump_indent();
    std::cout << "AssignStmt " << format_location(node);
    push_indent();
    node->get_lhs()->accept(this);
    set_last();
    node->get_rhs()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_return_stmt(return_stmt* node) {
    dump_indent();
    std::cout << "ReturnStmt " << format_location(node);
    push_indent();
    if (node->get_value()) {
        node->get_value()->accept(this);
    }
    pop_indent();
    return true;
}

bool dumper::visit_if_stmt(if_stmt* node) {
    dump_indent();
    std::cout << "IfStmt " << format_location(node);
    push_indent();
    node->get_condition()->accept(this);
    set_last();
    node->get_body()->accept(this);
    if (node->get_else_body()) {
        dump_indent();
        std::cout << "ElseStmt " << format_location(node->get_else_body());
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
    std::cout << "ForStmt " << node->get_iter() << format_location(node);
    push_indent();
    node->get_range()->accept(this);
    set_last();
    node->get_body()->accept(this);
    pop_indent();
    return true;
}

bool dumper::visit_block_stmt(block_stmt* node) {
    dump_indent();
    std::cout << "BlockStmt " << format_location(node);
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
    std::cout << "FuncDecl " << node->get_name() << format_location(node);
    push_indent();
    {
        dump_indent();
        std::cout << "Params:\n";
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
        std::cout << "ReturnType:\n";
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
    std::cout << "TypeDef " << node->get_base();
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
    std::cout << format_location(node);
    return true;
}

bool dumper::visit_param(param* node) {
    dump_indent();
    std::cout << "Param " << node->get_name() << format_location(node);
    push_indent();
    node->get_type()->accept(this);
    pop_indent();
    return true;
}

}
