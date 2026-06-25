#pragma once

#include "utils/type.hpp"

#include <iostream>
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
  TYPE_KIND_TUPLE
};

class type_impl {
protected:
    colgm_type_kind kind;

public:
    type_impl(colgm_type_kind kind) : kind(kind) {}
    virtual ~type_impl() = default;

    colgm_type_kind get_kind() const { return kind; }
    virtual std::string to_string() const { return ""; }
};

class type {
private:
    type_impl* impl;

protected:
    template <typename T>
    T* get_impl_as() const {
        return static_cast<T*>(impl);
    }

public:
    type(type_impl* impl) : impl(impl) {}

    colgm_type_kind get_kind() const { return impl->get_kind(); }
    auto get_impl() const { return impl; }
    bool operator==(const type& other) const { return impl == other.impl; }
    bool operator!=(const type& other) const { return !(*this == other); }
    std::string to_string() const { return impl->to_string(); }

    template <typename T>
    static bool isa(const type& t) {
        return T::is(t);
    }

    template <typename T>
    static const T& as(const type& t) {
        assert(type::isa<T>(t));
        return static_cast<const T&>(t);
    }
};

std::ostream& operator<<(std::ostream& os, const type& t);

class unknown_type_impl: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    unknown_type_impl(): type_impl(colgm_type_kind::TYPE_KIND_UNKNOWN) {}
    std::string to_string() const override { return "{unknown}"; }
};

class unknown_type: public type {
public:
    unknown_type(unknown_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_UNKNOWN;
    }
};

class int_type_impl: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    int_type_impl(i64 width):
        type_impl(width == 32 ?
                  colgm_type_kind::TYPE_KIND_INT32 :
                  colgm_type_kind::TYPE_KIND_INT64) {
        assert(width == 32 || width == 64);
    }
    std::string to_string() const override {
        return get_kind() == colgm_type_kind::TYPE_KIND_INT32 ? "i32" : "i64";
    }
};

class int_type: public type {
public:
    int_type(int_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_INT32 ||
               t.get_kind() == colgm_type_kind::TYPE_KIND_INT64;
    }
};

class float_type_impl: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    float_type_impl(i64 width):
        type_impl(width == 32 ?
                  colgm_type_kind::TYPE_KIND_FLOAT32 :
                  colgm_type_kind::TYPE_KIND_FLOAT64) {
        assert(width == 32 || width == 64);
    }
    std::string to_string() const override {
        return get_kind() == colgm_type_kind::TYPE_KIND_FLOAT32 ? "f32" : "f64";
    }
};

class float_type: public type {
public:
    float_type(float_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_FLOAT32 ||
               t.get_kind() == colgm_type_kind::TYPE_KIND_FLOAT64;
    }
};

class bool_type_impl: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    bool_type_impl(): type_impl(colgm_type_kind::TYPE_KIND_BOOL) {}
    std::string to_string() const override { return "bool"; }
};

class bool_type: public type {
public:
    bool_type(bool_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_BOOL;
    }
};

class void_type_impl: public type_impl {
public:
    using key_type = colgm_type_kind;

public:
    void_type_impl(): type_impl(colgm_type_kind::TYPE_KIND_VOID) {}
    std::string to_string() const override { return "void"; }
};

class void_type: public type {
public:
    void_type(void_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_VOID;
    }
};

class function_type_impl: public type_impl {
public:
    using key_type = std::tuple<std::vector<type>, type>;

private:
    std::vector<type> args;
    type ret;

public:
    function_type_impl(type ret):
        type_impl(colgm_type_kind::TYPE_KIND_FUNCTION), ret(ret) {}
    void add_argument(type arg) { args.push_back(arg); }
    const auto& get_arguments() const { return args; }
    auto get_return_type() const { return ret; }
    std::string to_string() const override;
};

class function_type: public type {
public:
    function_type(function_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION;
    }
    const auto& get_arguments() const {
        return get_impl_as<function_type_impl>()->get_arguments();
    }
    auto get_return_type() const {
        return get_impl_as<function_type_impl>()->get_return_type();
    }
};

class tensor_type_impl: public type_impl {
public:
    using key_type = std::tuple<type, std::vector<i64>>;

private:
    type element_type;
    std::vector<i64> shape;

public:
    tensor_type_impl(type et):
        type_impl(colgm_type_kind::TYPE_KIND_TENSOR), element_type(et) {}
    auto get_element_type() const { return element_type; }
    void add_dimension(i64 dim) { shape.push_back(dim); }
    const auto& get_shape() const { return shape; }
    std::string to_string() const override;
};

class tensor_type: public type {
public:
    tensor_type(tensor_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_TENSOR;
    }
    auto get_element_type() const {
        return get_impl_as<tensor_type_impl>()->get_element_type();
    }
    const auto& get_shape() const {
        return get_impl_as<tensor_type_impl>()->get_shape();
    }
};

class tuple_type_impl: public type_impl {
public:
    using key_type = std::vector<type>;

private:
    std::vector<type> types;

public:
    tuple_type_impl(): type_impl(colgm_type_kind::TYPE_KIND_TUPLE) {}
    void add_type(type t) { types.push_back(t); }
    const auto& get_types() const { return types; }
    std::string to_string() const override;
};

class tuple_type: public type {
public:
    tuple_type(tuple_type_impl* t): type(t) {}
    static bool is(const type& t) {
        return t.get_kind() == colgm_type_kind::TYPE_KIND_TUPLE;
    }
    const auto& get_types() const {
        return get_impl_as<tuple_type_impl>()->get_types();
    }
};

struct function_key_type_hash {
    usize operator()(const function_type_impl::key_type& key) const;
};

struct tensor_key_type_hash {
    usize operator()(const tensor_type_impl::key_type& key) const;
};

struct tuple_key_type_hash {
    usize operator()(const tuple_type_impl::key_type& key) const;
};

}
