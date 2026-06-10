#include "sema/intrinsic.hpp"

namespace colgm_mlir {

type relu_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "relu function takes one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "relu function takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (type::isa<tensor_type>(arg) ||
        type::isa<int_type>(arg) ||
        type::isa<float_type>(arg)) {
        return arg;
    }

    err.err(node->get_location(), "relu function takes only tensor, int or float argument");
    return ts.get_unknown_type();
}

}
