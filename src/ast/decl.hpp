#pragma once

#include "ast/ast.hpp"

namespace colgm_mlir {

class decl: public ast {
public:
    decl(ast::type t, span& loc): ast(t, loc) {}
    ~decl() override = default;
};

}
