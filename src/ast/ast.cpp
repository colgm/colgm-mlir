#include "ast/ast.hpp"
#include "ast/decl.hpp"
#include "ast/expr.hpp"
#include "ast/stmt.hpp"
#include "ast/visitor.hpp"

#include <cassert>

namespace colgm_mlir {

void ast::accept(visitor* v) {
    v->visit_ast(this);
    assert(false && "ast::accept() not implemented for this node type");
}

root::~root() {
    for (auto i : funcs) {
        if (i) {
            delete i;
        }
    }
}

void root::accept(visitor* v) {
    v->visit_root(this);
}

}