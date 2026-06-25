#include "sema/storage.hpp"

namespace colgm_mlir {

type_storage::~type_storage() {
    for (auto& [_, impl] : function_types) {
        delete impl;
    }
    for (auto& [_, impl] : tensor_types) {
        delete impl;
    }
    for (auto& [_, impl] : tuple_types) {
        delete impl;
    }
}

type type_storage::get_i32_type() {
    return get_tensor_type(i32_scalar_type_, {});
}

type type_storage::get_i64_type() {
    return get_tensor_type(i64_scalar_type_, {});
}

type type_storage::get_f32_type() {
    return get_tensor_type(f32_scalar_type_, {});
}

type type_storage::get_f64_type() {
    return get_tensor_type(f64_scalar_type_, {});
}

type type_storage::get_bool_type() {
    return get_tensor_type(bool_scalar_type_, {});
}

type type_storage::get_function_type(const std::vector<type>& args, type ret) {
    function_type_impl::key_type key(args, ret);
    auto [it, inserted] = function_types.insert({std::move(key), nullptr});
    if (!inserted) {
        return type(it->second);
    }

    auto impl = new function_type_impl(ret);
    for (auto& arg : args) {
        impl->add_argument(arg);
    }
    it->second = impl;
    return type(impl);
}

type type_storage::get_tensor_type(type et, const std::vector<i64>& dims) {
    tensor_type_impl::key_type key(et, dims);
    auto [it, inserted] = tensor_types.insert({std::move(key), nullptr});
    if (!inserted) {
        return type(it->second);
    }

    auto impl = new tensor_type_impl(et);
    for (auto& dim : dims) {
        impl->add_dimension(dim);
    }
    it->second = impl;
    return type(impl);
}

type type_storage::get_tuple_type(const std::vector<type>& types) {
    tuple_type_impl::key_type key(types);
    auto [it, inserted] = tuple_types.insert({std::move(key), nullptr});
    if (!inserted) {
        return type(it->second);
    }

    auto impl = new tuple_type_impl();
    for (auto& t : types) {
        impl->add_type(t);
    }
    it->second = impl;
    return type(impl);
}


}
