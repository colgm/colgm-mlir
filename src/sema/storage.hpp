#pragma once

#include "sema/type.hpp"

#include <unordered_map>
#include <vector>

namespace colgm_mlir {

class type_storage {
private:
    int_type i32_type_impl = int_type(32);
    int_type i64_type_impl = int_type(64);
    float_type f32_type_impl = float_type(32);
    float_type f64_type_impl = float_type(64);
    bool_type bool_type_impl;
    void_type void_type_impl;

    type i32_type_ = type(&i32_type_impl);
    type i64_type_ = type(&i64_type_impl);
    type f32_type_ = type(&f32_type_impl);
    type f64_type_ = type(&f64_type_impl);
    type bool_type_ = type(&bool_type_impl);
    type void_type_ = type(&void_type_impl);

    std::unordered_map<function_type::key_type,
                       function_type*,
                       function_key_type_hash> function_types;
    std::unordered_map<tensor_type::key_type,
                       tensor_type*,
                       tensor_key_type_hash> tensor_types;

public:
    ~type_storage();

    type get_i32_type() const { return i32_type_; }
    type get_i64_type() const { return i64_type_; }
    type get_f32_type() const { return f32_type_; }
    type get_f64_type() const { return f64_type_; }
    type get_bool_type() const { return bool_type_; }
    type get_void_type() const { return void_type_; }
    type get_function_type(const std::vector<type>&, type);
    type get_tensor_type(type, const std::vector<i64>&);
};

}
