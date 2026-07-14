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
    regist("print", print_infer);
    regist("matmul", matmul_infer);
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

type print_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() < 1) {
        err.warn(node->get_location(), "useless print");
        return ts.get_unknown_type();
    }

    return ts.get_void_type();
}

type matmul_infer(error& err, call_expr* node, type_storage& ts) {
    if (node->get_args().size() != 2) {
        err.err(node->get_location(), "matmul takes exactly two arguments");
        return ts.get_unknown_type();
    }

    auto lhs = node->get_args()[0]->get_resolved();
    auto rhs = node->get_args()[1]->get_resolved();

    if (!type::isa<tensor_type>(lhs) || !type::isa<tensor_type>(rhs)) {
        err.err(node->get_location(), "matmul takes only tensor arguments");
        return ts.get_unknown_type();
    }

    auto lhs_tt = type::as<tensor_type>(lhs);
    auto rhs_tt = type::as<tensor_type>(rhs);
    const auto& lhs_shape = lhs_tt.get_shape();
    const auto& rhs_shape = rhs_tt.get_shape();
    auto rank = lhs_shape.size();

    if (rank < 2 || rhs_shape.size() < 2) {
        err.err(node->get_location(), "matmul requires tensors with rank >= 2");
        return ts.get_unknown_type();
    }

    if (rank != rhs_shape.size()) {
        err.err(node->get_location(), "matmul tensors must have the same rank");
        return ts.get_unknown_type();
    }

    for (usize i = 0; i < rank - 2; i++) {
        if (lhs_shape[i] != rhs_shape[i]) {
            err.err(node->get_location(), "matmul batch dimensions must match");
            return ts.get_unknown_type();
        }
    }

    if (lhs_shape[rank - 1] != rhs_shape[rank - 2]) {
        err.err(node->get_location(), "matmul inner dimensions must match");
        return ts.get_unknown_type();
    }

    std::vector<i64> res_shape;
    for (usize i = 0; i < rank - 2; i++) {
        res_shape.push_back(lhs_shape[i]);
    }
    res_shape.push_back(lhs_shape[rank - 2]);
    res_shape.push_back(rhs_shape[rank - 1]);

    return ts.get_tensor_type(lhs_tt.get_element_type(), res_shape);
}

}
