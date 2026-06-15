#pragma once

#include "lexer/token.hpp"
#include "report/report.hpp"

#include "ast/ast.hpp"
#include "ast/decl.hpp"
#include "ast/expr.hpp"
#include "ast/stmt.hpp"

#include <vector>

namespace colgm_mlir {

class parser {
private:
    const std::vector<token>& tokens;
    error& err;
    u64 ptr = 0;
    root* tree = nullptr;

private:
    bool lookahead(tok t, i64 offset = 0) const;
    bool next_token_in_next_line() const {
        if (ptr + 1 < tokens.size()) {
            return tokens[ptr + 1].loc.begin_line > tokens[ptr].loc.begin_line;
        }
        return false;
    }

    bool this_token_in_next_line() const {
        if (ptr - 1 >= 0) {
            return tokens[ptr].loc.begin_line > tokens[ptr - 1].loc.begin_line;
        }
        return false;
    }
    void next() {
        if (tokens[ptr].type == tok::tk_eof) {
            return;
        }
        ptr++;
    }
    void match(tok t);
    void update_location(ast*);

private:
    type_def* parse_type();
    param* parse_param();
    func_decl* parse_func_decl();
    block_stmt* parse_block();
    var_decl* parse_var_decl();
    if_stmt* parse_if_stmt();
    for_stmt* parse_for_stmt();
    return_stmt* parse_return_stmt();
    stmt* parse_stmt();
    int_literal* parse_int_literal();
    float_literal* parse_float_literal();
    expr* parse_number();
    bool_literal* parse_bool_literal();
    tensor* parse_tensor();
    identifier* parse_identifier();
    range_expr* parse_range_expr();
    expr* parse_value();
    expr* parse_unary_operator();
    expr* parse_call_expr();
    expr* parse_index_access();
    expr* parse_cmp_expr();
    expr* parse_add_sub();
    expr* parse_mul_div();
    expr* parse_expr();

public:
    parser(const std::vector<token>& tokens, error& e):
        tokens(tokens), err(e) {}
    ~parser();
    auto get_tree() const { return tree; }
    const error& scan();
};

}
