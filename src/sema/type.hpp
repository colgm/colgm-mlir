#pragma once

#include "utils/type.hpp"

#include <cassert>
#include <vector>
#include <string>
#include <tuple>

namespace colgm_mlir {

enum class colgm_type_kind: u8 {
  TYPE_KIND_UNKNOWN = 0,
  TYPE_KIND_INT32,
  TYPE_KIND_INT64,
  TYPE_KIND_FLOAT32,
  TYPE_KIND_FLOAT64,
  TYPE_KIND_BOOL,
  TYPE_KIND_VOID,
  TYPE_KIND_FUNCTION,
  TYPE_KIND_TENSOR,
};

class type_impl {
protected:
    colgm_type_kind kind;

public:
    type_impl(colgm_type_kind kind) : kind(kind) {}
    virtual ~type_impl() = default;

    colgm_type_kind get_kind() const { return kind; }
};

class type {
private:
    type_impl* impl;

public:
    type(type_impl* impl) : impl(impl) {}

    colgm_type_kind get_kind() const { return impl->get_kind(); }
    auto get_impl() const { return impl; }
    bool operator==(const type& other) const { return impl == other.impl; }
    bool operator!=(const type& other) const { return !(*this == other); }
};

class int_type: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    int_type(i64 width):
        type_impl(width == 32 ?
                  colgm_type_kind::TYPE_KIND_INT32 :
                  colgm_type_kind::TYPE_KIND_INT64) {}
};

class float_type: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    float_type(i64 width):
        type_impl(width == 32 ?
                  colgm_type_kind::TYPE_KIND_FLOAT32 :
                  colgm_type_kind::TYPE_KIND_FLOAT64) {}
};

class bool_type: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    bool_type(): type_impl(colgm_type_kind::TYPE_KIND_BOOL) {}
};

class void_type: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    void_type(): type_impl(colgm_type_kind::TYPE_KIND_VOID) {}
};

class function_type: public type_impl {
public:
    using key_type = std::tuple<std::vector<type>, type>;

private:
    std::vector<type> args;
    type ret;

public:
    function_type(type ret):
        type_impl(colgm_type_kind::TYPE_KIND_FUNCTION), ret(ret) {}

    void add_argument(type arg) { args.push_back(arg); }

    const auto& get_arguments() const { return args; }

    auto get_return_type() const { return ret; }
};

class tensor_type: public type_impl {
public:
    using key_type = std::tuple<type, std::vector<i64>>;

private:
    type element_type;
    std::vector<i64> shape;

public:
    tensor_type(type et):
        type_impl(colgm_type_kind::TYPE_KIND_TENSOR), element_type(et) {}

    auto get_element_type() const { return element_type; }

    void add_dimension(i64 dim) { shape.push_back(dim); }

    const auto& get_shape() const { return shape; }
};

struct function_key_type_hash {
    size_t operator()(const function_type::key_type& key) const;
};

struct tensor_key_type_hash {
    size_t operator()(const tensor_type::key_type& key) const;
};

}
