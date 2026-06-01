#include "parse/parser.hpp"

namespace colgm_mlir {

parser::~parser() {
    if (tree) {
        delete tree;
    }
}

bool parser::lookahead(tok t) const {
    return tokens[ptr].type == t;
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

type_def* parser::parse_type() {
    auto node = new type_def(tokens[ptr].loc);
    node->set_base(tokens[ptr].str);
    match(tok::tk_id);
    if (lookahead(tok::tk_lbracket)) {
        match(tok::tk_lbracket);
        while (!lookahead(tok::tk_rbracket)) {
            // TODO
            next();
        }
        match(tok::tk_rbracket);
    }
    return node;
}

param* parser::parse_param() {
    auto node = new param(tokens[ptr].loc);
    node->set_name(tokens[ptr].str);
    match(tok::tk_id);
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
    return node;
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

    return err;
}

}
