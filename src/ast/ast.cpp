#include "ast/ast.hpp"
#include "ast/visitor.hpp"

#include <cassert>

namespace colgm_mlir {

void ast::accept(visitor*) {
    assert(false && "ast::accept() not implemented for this node type");
}

}