#include "sema/intrinsic.hpp"

namespace colgm_mlir {

intrinsic_registry::intrinsic_registry() {
    regist("relu", relu_infer);
    regist("abs", abs_infer);
    regist("exp", exp_infer);
    regist("log", log_infer);
    regist("sqrt", sqrt_infer);
    regist("tanh", tanh_infer);
    regist("sigmoid", sigmoid_infer);
}

intrinsic_find_res intrinsic_registry::find(const std::string& name) const {
    if (intrinsics.find(name) != intrinsics.end()) {
        return intrinsic_find_res { intrinsics.at(name), true };
    }
    return intrinsic_find_res { nullptr, false };
}

static bool is_calculation_type(const type& t) {
    if (!type::isa<tensor_type>(t)) {
        return false;
    }

    auto tt = type::as<tensor_type>(t);
    return type::isa<int_type>(tt.get_element_type()) ||
           type::isa<float_type>(tt.get_element_type());
}

static bool is_float_type(const type& t) {
    if (!type::isa<tensor_type>(t)) {
        return false;
    }

    auto tt = type::as<tensor_type>(t);
    return type::isa<float_type>(tt.get_element_type());
}

type relu_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "relu takes at least one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "relu takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (is_calculation_type(arg)) {
        return arg;
    }

    err.err(node->get_location(), "relu takes only tensor, int or float argument");
    return ts.get_unknown_type();
}

type abs_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "abs takes at least one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "abs takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (is_calculation_type(arg)) {
        return arg;
    }

    err.err(node->get_location(), "abs takes only tensor, int or float argument");
    return ts.get_unknown_type();
}

type exp_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "exp takes at least one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "exp takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (is_float_type(arg)) {
        return arg;
    }

    err.err(node->get_location(), "exp takes only tensor, float argument");
    return ts.get_unknown_type();
}

type log_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "log takes at least one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "log takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (is_float_type(arg)) {
        return arg;
    }

    err.err(node->get_location(), "log takes only tensor, float argument");
    return ts.get_unknown_type();
}

type sqrt_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "sqrt takes at least one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "sqrt takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (is_float_type(arg)) {
        return arg;
    }

    err.err(node->get_location(), "sqrt takes only tensor, float argument");
    return ts.get_unknown_type();
}

type tanh_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "tanh takes at least one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "tanh takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (is_float_type(arg)) {
        return arg;
    }

    err.err(node->get_location(), "tanh takes only tensor, float argument");
    return ts.get_unknown_type();
}

type sigmoid_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.err(node->get_location(), "sigmoid takes at least one argument");
        return ts.get_unknown_type();
    } else if (node->get_args().size() > 1) {
        err.err(node->get_location(), "sigmoid takes only one argument");
        return ts.get_unknown_type();
    }

    auto arg = node->get_args()[0]->get_resolved();
    if (is_float_type(arg)) {
        return arg;
    }

    err.err(node->get_location(), "sigmoid takes only tensor, float argument");
    return ts.get_unknown_type();
}

}
