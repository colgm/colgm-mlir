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
    bool lookahead(tok t) const;
    void next() {
        if (tokens[ptr].type == tok::tk_eof) {
            return;
        }
        ptr++;
    }

public:
    parser(const std::vector<token>& tokens, error& e):
        tokens(tokens), err(e) {}
    ~parser();
    auto get_tree() const { return tree; }
    const error& scan();
};

}
