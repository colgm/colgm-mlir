#include "ast/expr.hpp"

namespace colgm_mlir {

void expr::accept(visitor* v) {
    v->visit_expr(this);
}

void int_literal::accept(visitor* v) {
    v->visit_int_literal(this);
}

void float_literal::accept(visitor* v) {
    v->visit_float_literal(this);
}

void bool_literal::accept(visitor* v) {
    v->visit_bool_literal(this);
}

tensor::~tensor() {
    for (auto& i : values) {
        if (i) {
            delete i;
        }
    }
}

void tensor::accept(visitor* v) {
    v->visit_tensor(this);
}

void identifier::accept(visitor* v) {
    v->visit_identifier(this);
}

binary_expr::~binary_expr() {
    if (lhs) {
        delete lhs;
    }
    if (rhs) {
        delete rhs;
    }
}

void binary_expr::accept(visitor* v) {
    v->visit_binary_expr(this);
}

unary_expr::~unary_expr() {
    if (operand) {
        delete operand;
    }
}

void unary_expr::accept(visitor* v) {
    v->visit_unary_expr(this);
}

call_expr::~call_expr() {
    if (callee) {
        delete callee;
    }
    for (auto i : args) {
        if (i) {
            delete i;
        }
    }
}

void call_expr::accept(visitor* v) {
    v->visit_call_expr(this);
}

index_access::~index_access() {
    if (target) {
        delete target;
    }
    if (index) {
        delete index;
    }
}

void index_access::accept(visitor* v) {
    v->visit_index_access(this);
}

range_expr::~range_expr() {
    if (start) {
        delete start;
    }
    if (end) {
        delete end;
    }
}

void range_expr::accept(visitor* v) {
    v->visit_range_expr(this);
}

if_expr::~if_expr() {
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

void if_expr::accept(visitor* v) {
    v->visit_if_expr(this);
}

for_expr::~for_expr() {
    if (range) {
        delete range;
    }
    for (const auto& i : init_pairs) {
        auto v = std::get<1>(i);
        if (v) {
            delete v;
        }
    }
    if (body) {
        delete body;
    }
}

void for_expr::accept(visitor* v) {
    v->visit_for_expr(this);
}

}
