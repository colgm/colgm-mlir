#include "ast/decl.hpp"
#include "ast/stmt.hpp"

namespace colgm_mlir {

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

param::~param() {
    if (type) {
        delete type;
    }
}

}
