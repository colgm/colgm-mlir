#pragma once

#include "ast/ast.hpp"
#include "ast/visitor.hpp"

#include <vector>

namespace colgm_mlir {

class decl: public ast {
public:
    decl(ast_type t, const span& loc): ast(t, loc) {}
    ~decl() override = default;
    void accept(visitor*) override;
};

class func_decl: public decl {
private:
    std::string name;
    std::vector<param*> params;
    type_def* return_type = nullptr;
    block_stmt* body = nullptr;

public:
    func_decl(const span& loc): decl(ast_type::func_decl, loc) {}
    ~func_decl() override;
    void accept(visitor*) override;
    void set_name(const std::string& n) { name = n; }
    const auto& get_name() const { return name; }
    void add_param(param* p) { params.push_back(p); }
    const auto& get_params() const { return params; }
    void set_return_type(type_def* t) { return_type = t; }
    auto get_return_type() const { return return_type; }
    void set_body(block_stmt* b) { body = b; }
    auto get_body() const { return body; }
};

class type_def: public decl {
private:
    std::string base;
    std::vector<i64> dims;

public:
    type_def(const span& loc): decl(ast_type::type_def, loc) {}
    ~type_def() override = default;
    void accept(visitor*) override;
    void set_base(const std::string& b) { base = b; }
    const auto& get_base() const { return base; }
    void add_dim(i64 d) { dims.push_back(d); }
    const auto& get_dims() const { return dims; }
    bool is_scalar() const { return dims.empty(); }
};

class param: public decl {
private:
    std::string name;
    type_def* type = nullptr;

public:
    param(const span& loc): decl(ast_type::param, loc) {}
    ~param() override;
    void accept(visitor*) override;
    void set_name(const std::string& n) { name = n; }
    const auto& get_name() const { return name; }
    void set_type(type_def* t) { type = t; }
    auto get_type() const { return type; }
};

}
