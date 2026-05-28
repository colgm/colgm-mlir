#include "ast/ast.hpp"
#include "ast/visitor.hpp"

#include <cassert>

namespace colgm_mlir {

void ast::accept(visitor* v) {
    v->visit_ast(this);
    assert(false && "ast::accept() not implemented for this node type");
}

}