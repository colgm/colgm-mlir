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

public:
    parser(const std::vector<token>& tokens, error& e):
        tokens(tokens), err(e) {}
};

}
