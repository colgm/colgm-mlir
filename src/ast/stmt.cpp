#include "ast/stmt.hpp"
#include "ast/expr.hpp"

namespace colgm_mlir {

var_decl::~var_decl() {
    if (init) {
        delete init;
    }
}

assign_stmt::~assign_stmt() {
    if (lhs) {
        delete lhs;
    }
    if (rhs) {
        delete rhs;
    }
}

return_stmt::~return_stmt() {
    if (value) {
        delete value;
    }
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

for_stmt::~for_stmt() {
    if (range) {
        delete range;
    }
    if (body) {
        delete body;
    }
}

block_stmt::~block_stmt() {
    for (auto i : stmts) {
        if (i) {
            delete i;
        }
    }
}

}
