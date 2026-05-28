#include "ast/decl.hpp"
#include "ast/stmt.hpp"

namespace colgm_mlir {

void decl::accept(visitor* v) {
    v->visit_decl(this);
}

func_decl::~func_decl() {
    for (auto i : params) {
        if (i) {
            delete i;
        }
    }

    if (return_type) {
        delete return_type;
    }

    if (body) {
        delete body;
    }
}

void func_decl::accept(visitor* v) {
    v->visit_func_decl(this);
}

void type_def::accept(visitor* v) {
    v->visit_type_def(this);
}

param::~param() {
    if (type) {
        delete type;
    }
}

void param::accept(visitor* v) {
    v->visit_param(this);
}

}
