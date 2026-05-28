#include "ast/expr.hpp"

namespace colgm_mlir {

binary_expr::~binary_expr() {
    if (lhs) {
        delete lhs;
    }
    if (rhs) {
        delete rhs;
    }
}

unary_expr::~unary_expr() {
    if (operand) {
        delete operand;
    }
}

call_expr::~call_expr() {
    if (callee) {
        delete callee;
    }
    for (auto& i : args) {
        if (i.value) {
            delete i.value;
            i.value = nullptr;
        }
    }
}

index_access::~index_access() {
    if (target) {
        delete target;
    }
    if (index) {
        delete index;
    }
}

range_expr::~range_expr() {
    if (start) {
        delete start;
    }
    if (end) {
        delete end;
    }
}

}
