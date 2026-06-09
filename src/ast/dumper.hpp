#pragma once

#include "ast/visitor.hpp"
#include "utils/colorful.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>

namespace colgm_mlir {

struct fmt_helper {
    ast* node;

    friend std::ostream& operator<<(std::ostream& os, const fmt_helper& f) {
        if (f.node->get_resolved().get_impl()) {
            os << cyan << " [" << f.node->get_resolved() << "]" << reset;
        }
        os << white << " -> " << f.node->get_location() << reset << "\n";
        return os;
    }
};

class dumper: public visitor {
private:
    std::vector<std::string> indent;

private:
    void push_indent() {
        if (indent.size()) {
            if (indent.back()=="|--") {
                indent.back() = "|  ";
            } else if (indent.back()=="`--") {
                indent.back() = "   ";
            }
        }
        indent.push_back("|--");
    }
    void pop_indent() { indent.pop_back(); }
    void set_last() { indent.back() = "`--"; }
    void dump_indent() {
        if (indent.size() && indent.back()=="|  ") {
            indent.back() = "|--";
        }
        for (const auto& i : indent) {
            std::cout << i;
        }
    }
    fmt_helper format_info(ast* n) {
        return fmt_helper { n };
    }

private:
    bool visit_root(root*) override;

    bool visit_int_literal(int_literal*) override;
    bool visit_float_literal(float_literal*) override;
    bool visit_bool_literal(bool_literal*) override;
    bool visit_tensor(tensor*) override;
    bool visit_identifier(identifier*) override;
    bool visit_binary_expr(binary_expr*) override;
    bool visit_unary_expr(unary_expr*) override;
    bool visit_call_expr(call_expr*) override;
    bool visit_index_access(index_access*) override;
    bool visit_range_expr(range_expr*) override;

    bool visit_var_decl(var_decl*) override;
    bool visit_assign_stmt(assign_stmt*) override;
    bool visit_return_stmt(return_stmt*) override;
    bool visit_if_stmt(if_stmt*) override;
    bool visit_for_stmt(for_stmt*) override;
    bool visit_block_stmt(block_stmt*) override;

    bool visit_func_decl(func_decl*) override;
    bool visit_type_def(type_def*) override;
    bool visit_param(param*) override;

public:
    dumper(): indent({}) {}
    static void dump(ast* n) {
        dumper d;
        n->accept(&d);
    }
};

}