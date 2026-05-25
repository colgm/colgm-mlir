#pragma once

#include "ast/ast.hpp"

namespace colgm_mlir {

class expr: public ast {
public:
    expr(ast::type t, span& loc): ast(t, loc) {}
    ~expr() override = default;
};

}
