#pragma once

#include "sema/type.hpp"

#include <unordered_map>
#include <vector>

namespace colgm_mlir {

class type_storage {
private:
    unknown_type_impl unknown_type_impl_;
    int_type_impl i32_type_impl_ = int_type_impl(32);
    int_type_impl i64_type_impl_ = int_type_impl(64);
    float_type_impl f32_type_impl_ = float_type_impl(32);
    float_type_impl f64_type_impl_ = float_type_impl(64);
    bool_type_impl bool_type_impl_;
    void_type_impl void_type_impl_;

    type unknown_type_ = type(&unknown_type_impl_);
    type i32_type_ = type(&i32_type_impl_);
    type i64_type_ = type(&i64_type_impl_);
    type f32_type_ = type(&f32_type_impl_);
    type f64_type_ = type(&f64_type_impl_);
    type bool_type_ = type(&bool_type_impl_);
    type void_type_ = type(&void_type_impl_);

    std::unordered_map<function_type_impl::key_type,
                       function_type_impl*,
                       function_key_type_hash> function_types;
    std::unordered_map<tensor_type_impl::key_type,
                       tensor_type_impl*,
                       tensor_key_type_hash> tensor_types;

public:
    ~type_storage();
    type get_unknown_type() const { return unknown_type_; }
    type get_i32_type() const { return i32_type_; }
    type get_i64_type() const { return i64_type_; }
    type get_f32_type() const { return f32_type_; }
    type get_f64_type() const { return f64_type_; }
    type get_bool_type() const { return bool_type_; }
    type get_void_type() const { return void_type_; }
    type get_function_type(const std::vector<type>&, type);
    type get_tensor_type(type, const std::vector<i64>&);

    const auto& get_function_type_cache() const { return function_types; }
    const auto& get_tensor_type_cache() const { return tensor_types; }
};

}
