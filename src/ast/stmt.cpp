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

assign_stmt::~assign_stmt() {
    if (lhs) {
        delete lhs;
    }
    if (rhs) {
        delete rhs;
    }
}

void assign_stmt::accept(visitor* v) {
    v->visit_assign_stmt(this);
}

return_stmt::~return_stmt() {
    if (value) {
        delete value;
    }
}

void return_stmt::accept(visitor* v) {
    v->visit_return_stmt(this);
}

if_stmt::~if_stmt() {
    if (condition) {
        delete condition;
    }
    if (body) {
        delete body;
    }
    if (else_body) {
        delete else_body;
    }
}

void if_stmt::accept(visitor* v) {
    v->visit_if_stmt(this);
}

for_stmt::~for_stmt() {
    if (range) {
        delete range;
    }
    if (body) {
        delete body;
    }
}

void for_stmt::accept(visitor* v) {
    v->visit_for_stmt(this);
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
