#include "ast/stmt.hpp"
#include "ast/expr.hpp"

namespace colgm_mlir {

void stmt::accept(visitor* v) {
    v->visit_stmt(this);
}

var_decl::~var_decl() {
    if (init) {
        delete init;
    }
}

void var_decl::accept(visitor* v) {
    v->visit_var_decl(this);
}

return_stmt::~return_stmt() {
    if (value) {
        delete value;
    }
}

void return_stmt::accept(visitor* v) {
    v->visit_return_stmt(this);
}

yield_stmt::~yield_stmt() {
    if (value) {
        delete value;
    }
}

void yield_stmt::accept(visitor* v) {
    v->visit_yield_stmt(this);
}

expr_stmt::~expr_stmt() {
    if (inner) {
        delete inner;
    }
}

void expr_stmt::accept(visitor* v) {
    v->visit_expr_stmt(this);
}

block_stmt::~block_stmt() {
    for (auto i : stmts) {
        if (i) {
            delete i;
        }
    }
}

void block_stmt::accept(visitor* v) {
    v->visit_block_stmt(this);
}

}
