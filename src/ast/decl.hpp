#pragma once

#include "ast/ast.hpp"

#include <vector>

namespace colgm_mlir {

class decl: public ast {
public:
    decl(ast::type t, const span& loc): ast(t, loc) {}
    ~decl() override = default;
};

class func_decl: public decl {
private:
    std::vector<param*> params;
    type_def* return_type = nullptr;
    block_stmt* body = nullptr;

public:
    func_decl(const span& loc): decl(ast::type::func_decl, loc) {}
    void add_param(param* p) { params.push_back(p); }
    const auto& get_params() const { return params; }
    void set_return_type(type_def* t) { return_type = t; }
    auto get_return_type() const { return return_type; }
};

class type_def: public decl {
public:
    enum class dtype_kind {
        t_f32,
        t_f64,
        t_i32,
        t_i64,
        t_bool
    };

private:
    dtype_kind dtype = dtype_kind::t_i32;
    std::vector<i64> dims;

public:
    type_def(const span& loc): decl(ast::type::type_def, loc) {}
    void set_dtype(dtype_kind k) { dtype = k; }
    auto get_dtype() const { return dtype; }
    void add_dim(i64 d) { dims.push_back(d); }
    const auto& get_dims() const { return dims; }
    bool is_scalar() const { return dims.empty(); }
};

class param: public decl {
private:
    std::string name;
    type_def* type = nullptr;

public:
    param(const span& loc): decl(ast::type::param, loc) {}
    void set_name(const std::string& n) { name = n; }
    const auto& get_name() const { return name; }
    void set_type(type_def* t) { type = t; }
    auto get_type() const { return type; }
};

}
