#include "utils/colgm.hpp"
#include "parse/parser.hpp"

namespace colgm_mlir {

parser::~parser() {
    if (tree) {
        delete tree;
    }
}

bool parser::lookahead(tok t, i64 offset) const {
    return tokens[ptr + offset].type == t;
}

void parser::match(tok t) {
    if (tokens[ptr].type == t) {
        next();
        return;
    }
    err.err(
        tokens[ptr].loc,
        "Expected '" + std::string(token::to_string(t)) +
        "' but got '" + tokens[ptr].type_to_string() + "'"
    );
    next();
}

void parser::update_location(ast* node) {
    const token& end = (ptr >= 1) ? tokens[ptr - 1] : tokens[ptr];
    node->update_location(end.loc);
}

type_def* parser::parse_type() {
    auto node = new type_def(tokens[ptr].loc);
    node->set_base(tokens[ptr].str);
    match(tok::tk_id);
    if (lookahead(tok::tk_lbracket)) {
        match(tok::tk_lbracket);
        while (!lookahead(tok::tk_rbracket)) {
            // TODO
            auto num = dec_to_u64(tokens[ptr].str.c_str());
            node->add_dim(num);
            match(tok::tk_num);
            if (lookahead(tok::tk_comma)) {
                match(tok::tk_comma);
            } else if (lookahead(tok::tk_num)) {
                err.err(tokens[ptr - 1].loc, "Expect ',' after this");
            }
        }
        match(tok::tk_rbracket);
    }
    update_location(node);
    return node;
}

param* parser::parse_param() {
    auto node = new param(tokens[ptr].loc);
    node->set_name(tokens[ptr].str);
    match(tok::tk_id);
    if (lookahead(tok::tk_colon)) {
        match(tok::tk_colon);
        node->set_type(parse_type());
    }
    update_location(node);
    return node;
}

func_decl* parser::parse_func_decl() {
    auto node = new func_decl(tokens[ptr].loc);
    match(tok::tk_func);
    node->set_name(tokens[ptr].str);
    match(tok::tk_id);
    match(tok::tk_lparen);
    while (!lookahead(tok::tk_rparen)) {
        node->add_param(parse_param());
        if (lookahead(tok::tk_comma)) {
            match(tok::tk_comma);
        } else if (!lookahead(tok::tk_rparen)) {
            err.err(tokens[ptr].loc, "Expected ',' or ')'");
        }
    }
    match(tok::tk_rparen);

    if (lookahead(tok::tk_arrow)) {
        match(tok::tk_arrow);
        node->set_return_type(parse_type());
    }

    node->set_body(parse_block());

    update_location(node);
    return node;
}

block_stmt* parser::parse_block() {
    auto node = new block_stmt(tokens[ptr].loc);
    if (!lookahead(tok::tk_lbrace)) {
        auto stmt = parse_stmt();
        if (stmt) {
            node->add_stmt(stmt);
        }
        update_location(node);
        return node;
    }
    match(tok::tk_lbrace);
    while (!lookahead(tok::tk_rbrace)) {
        auto stmt = parse_stmt();
        if (stmt) {
            node->add_stmt(stmt);
        }
        if (lookahead(tok::tk_eof)) {
            break;
        }
        while (lookahead(tok::tk_semi)) {
            match(tok::tk_semi);
        }
    }
    match(tok::tk_rbrace);
    update_location(node);
    return node;
}

stmt* parser::parse_stmt() {
    if (lookahead(tok::tk_var)) {
        return parse_var_decl();
    } else if (lookahead(tok::tk_if)) {
        return parse_if_stmt();
    } else if (lookahead(tok::tk_for)) {
        return parse_for_stmt();
    } else if (lookahead(tok::tk_ret)) {
        return parse_return_stmt();
    } else if (lookahead(tok::tk_yield)) {
        return parse_yield_stmt();
    } else {
        err.err(tokens[ptr].loc,
            "Unexpected statement token '" + tokens[ptr].str + "'"
        );
        next();
    }
    return nullptr;
}

var_decl* parser::parse_var_decl() {
    auto node = new var_decl(tokens[ptr].loc);
    match(tok::tk_var);
    node->set_name(tokens[ptr].str);
    match(tok::tk_id);
    match(tok::tk_eq);
    node->set_init(parse_expr());
    update_location(node);
    return node;
}

if_stmt* parser::parse_if_stmt() {
    auto node = new if_stmt(tokens[ptr].loc);
    match(tok::tk_if);
    node->set_condition(parse_expr());
    node->set_body(parse_block());
    if (lookahead(tok::tk_else)) {
        match(tok::tk_else);
        node->set_else_body(parse_block());
    }
    update_location(node);
    return node;
}

for_stmt* parser::parse_for_stmt() {
    auto node = new for_stmt(tokens[ptr].loc);
    match(tok::tk_for);
    node->set_iter(tokens[ptr].str);
    match(tok::tk_id);
    match(tok::tk_in);
    node->set_range(parse_range_expr());
    node->set_body(parse_block());
    update_location(node);
    return node;
}

return_stmt* parser::parse_return_stmt() {
    auto node = new return_stmt(tokens[ptr].loc);
    match(tok::tk_ret);
    if (lookahead(tok::tk_semi) || this_token_in_next_line()) {
        return node;
    }

    if (lookahead(tok::tk_id) ||
        lookahead(tok::tk_num) ||
        lookahead(tok::tk_true) ||
        lookahead(tok::tk_false) ||
        lookahead(tok::tk_add) ||
        lookahead(tok::tk_sub) ||
        lookahead(tok::tk_lparen) ||
        lookahead(tok::tk_lbracket)) {
        node->set_value(parse_expr());
    }
    update_location(node);
    return node;
}

yield_stmt* parser::parse_yield_stmt() {
    auto node = new yield_stmt(tokens[ptr].loc);
    match(tok::tk_yield);
    if (lookahead(tok::tk_semi) || this_token_in_next_line()) {
        return node;
    }

    if (lookahead(tok::tk_id) ||
        lookahead(tok::tk_num) ||
        lookahead(tok::tk_true) ||
        lookahead(tok::tk_false) ||
        lookahead(tok::tk_add) ||
        lookahead(tok::tk_sub) ||
        lookahead(tok::tk_lparen) ||
        lookahead(tok::tk_lbracket)) {
        node->set_value(parse_expr());
    }
    update_location(node);
    return node;
}

range_expr* parser::parse_range_expr() {
    auto node = new range_expr(tokens[ptr].loc);
    node->set_start(parse_expr());
    match(tok::tk_dot);
    match(tok::tk_dot);
    node->set_end(parse_expr());

    update_location(node);
    return node;
}

int_literal* parser::parse_int_literal() {
    auto node = new int_literal(tokens[ptr].loc);
    node->set_literal(dec_to_u64(tokens[ptr].str.c_str()));
    match(tok::tk_num);
    return node;
}

float_literal* parser::parse_float_literal() {
    auto node = new float_literal(tokens[ptr].loc);
    node->set_literal(dec_to_f64(tokens[ptr].str.c_str()));
    match(tok::tk_num);
    return node;
}

expr* parser::parse_number() {
    if (tokens[ptr].str.find('.') != std::string::npos ||
        tokens[ptr].str.find('e') != std::string::npos ||
        tokens[ptr].str.find('E') != std::string::npos) {
        return parse_float_literal();
    } else {
        return parse_int_literal();
    }
}

bool_literal* parser::parse_bool_literal() {
    auto node = new bool_literal(tokens[ptr].loc);
    node->set_flag(lookahead(tok::tk_true));
    next();
    return node;
}

tensor* parser::parse_tensor() {
    auto node = new tensor(tokens[ptr].loc);
    match(tok::tk_lbracket);
    while (!lookahead(tok::tk_rbracket)) {
        node->add_value(parse_expr());
        if (lookahead(tok::tk_comma)) {
            match(tok::tk_comma);
        } else if (!lookahead(tok::tk_rbracket)) {
            err.err(tokens[ptr].loc, "Expected ',' or ']'");
        }
    }
    match(tok::tk_rbracket);

    update_location(node);
    return node;
}

identifier* parser::parse_identifier() {
    auto node = new identifier(tokens[ptr].loc);
    node->set_name(tokens[ptr].str);
    match(tok::tk_id);
    return node;
}

expr* parser::parse_value() {
    if (lookahead(tok::tk_num)) {
        return parse_number();
    } else if (lookahead(tok::tk_true) || lookahead(tok::tk_false)) {
        return parse_bool_literal();
    } else if (lookahead(tok::tk_id)) {
        if (lookahead(tok::tk_lparen, 1)) {
            return parse_call_expr();
        } else if (lookahead(tok::tk_lbracket, 1)) {
            return parse_index_access();
        } else {
            return parse_identifier();
        }
    } else if (lookahead(tok::tk_lbracket)) {
        return parse_tensor();
    } else if (lookahead(tok::tk_lparen)) {
        match(tok::tk_lparen);
        auto node = parse_expr();
        match(tok::tk_rparen);
        return node;
    } else if (lookahead(tok::tk_sub)) {
        return parse_unary_operator();
    }

    err.err(tokens[ptr].loc, "Expected a value");
    next();
    return nullptr;
}

expr* parser::parse_unary_operator() {
    auto node = new unary_expr(tokens[ptr].loc);
    if (lookahead(tok::tk_sub)) {
        node->set_op_type(unary_expr::op::sub);
        next();
    }
    node->set_operand(parse_value());
    update_location(node);
    return node;
}

expr* parser::parse_call_expr() {
    auto callee = parse_identifier();
    if (!lookahead(tok::tk_lparen)) {
        return callee;
    }

    auto node = new call_expr(tokens[ptr].loc);
    node->set_callee(callee);
    match(tok::tk_lparen);
    while (!lookahead(tok::tk_rparen)) {
        node->add_arg(parse_expr());
        if (lookahead(tok::tk_comma)) {
            match(tok::tk_comma);
        } else if (!lookahead(tok::tk_rparen)) {
            err.err(tokens[ptr].loc, "Expected ',' or ')'");
        }
    }
    match(tok::tk_rparen);
    update_location(node);
    return node;
}

expr* parser::parse_index_access() {
    auto callee = parse_identifier();
    expr* node = callee;
    while (lookahead(tok::tk_lbracket)) {
        auto res = new index_access(tokens[ptr].loc);
        res->set_target(node);
        match(tok::tk_lbracket);
        res->set_index(parse_expr());
        match(tok::tk_rbracket);
        update_location(res);
        node = res;
    }
    return node;
}

expr* parser::parse_cmp_expr() {
    auto lhs = parse_add_sub();
    if (lookahead(tok::tk_cmpeq) ||
        lookahead(tok::tk_neq) ||
        lookahead(tok::tk_less) ||
        lookahead(tok::tk_leq) ||
        lookahead(tok::tk_grt) ||
        lookahead(tok::tk_geq)) {
        auto node = new binary_expr(tokens[ptr].loc);
        switch (tokens[ptr].type) {
            case tok::tk_cmpeq: node->set_op_type(binary_expr::op::cmp_eq); break;
            case tok::tk_neq: node->set_op_type(binary_expr::op::cmp_ne); break;
            case tok::tk_less: node->set_op_type(binary_expr::op::cmp_lt); break;
            case tok::tk_leq: node->set_op_type(binary_expr::op::cmp_le); break;
            case tok::tk_grt: node->set_op_type(binary_expr::op::cmp_gt); break;
            case tok::tk_geq: node->set_op_type(binary_expr::op::cmp_ge); break;
            default: break;
        }
        next();
        node->set_lhs(lhs);
        node->set_rhs(parse_add_sub());
        update_location(node);
        return node;
    }
    return lhs;
}

expr* parser::parse_add_sub() {
    auto lhs = parse_mul_div();
    while (lookahead(tok::tk_add) || lookahead(tok::tk_sub)) {
        auto node = new binary_expr(tokens[ptr].loc);
        if (lookahead(tok::tk_add)) {
            node->set_op_type(binary_expr::op::add);
            next();
        } else if (lookahead(tok::tk_sub)) {
            node->set_op_type(binary_expr::op::sub);
            next();
        }
        node->set_lhs(lhs);
        node->set_rhs(parse_mul_div());
        lhs = node;
        update_location(lhs);
    }
    return lhs;
}

expr* parser::parse_mul_div() {
    auto lhs = parse_value();
    while (lookahead(tok::tk_mul) || lookahead(tok::tk_div)) {
        auto node = new binary_expr(tokens[ptr].loc);
        if (lookahead(tok::tk_mul)) {
            node->set_op_type(binary_expr::op::mul);
            next();
        } else if (lookahead(tok::tk_div)) {
            node->set_op_type(binary_expr::op::div);
            next();
        }
        node->set_lhs(lhs);
        node->set_rhs(parse_value());
        lhs = node;
        update_location(lhs);
    }
    return lhs;
}

expr* parser::parse_expr() {
    return parse_cmp_expr();
}

const error& parser::scan() {
    tree = new root(tokens[0].loc);
    ptr = 0;
    while (!lookahead(tok::tk_eof)) {
        if (lookahead(tok::tk_func)) {
            tree->add_func(parse_func_decl());
            continue;
        }
        err.err(tokens[ptr].loc, "Unexpected token '" + tokens[ptr].type_to_string() + "'");
        while (!lookahead(tok::tk_func) && !lookahead(tok::tk_eof)) {
            next();
        }
    }

    update_location(tree);
    return err;
}

}
