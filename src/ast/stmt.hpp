#pragma once

#include "ast/ast.hpp"

#include <vector>
#include <string>

namespace colgm_mlir {

class stmt: public ast {
public:
    stmt(ast::type t, const span& loc): ast(t, loc) {}
    ~stmt() override = default;
};

class var_decl: public stmt {
private:
    std::string name;
    expr* init = nullptr;

public:
    var_decl(const span& loc): stmt(ast::type::var_decl, loc) {}
    ~var_decl() override;
    void set_name(const std::string& n) { name = n; }
    const auto& get_name() const { return name; }
    void set_init(expr* i) { init = i; }
    auto get_init() const { return init; }
};

class assign_stmt: public stmt {
private:
    expr* lhs = nullptr;
    expr* rhs = nullptr;

public:
    assign_stmt(const span& loc): stmt(ast::type::assign_stmt, loc) {}
    ~assign_stmt() override;
    void set_lhs(expr* l) { lhs = l; }
    auto get_lhs() const { return lhs; }
    void set_rhs(expr* r) { rhs = r; }
    auto get_rhs() const { return rhs; }
};

class return_stmt: public stmt {
private:
    expr* value = nullptr;

public:
    return_stmt(const span& loc): stmt(ast::type::return_stmt, loc) {}
    ~return_stmt() override;
    void set_value(expr* v) { value = v; }
    auto get_value() const { return value; }
};

class if_stmt: public stmt {
private:
    expr* condition = nullptr;
    stmt* body = nullptr;
    stmt* else_body = nullptr;

public:
    if_stmt(const span& loc): stmt(ast::type::if_stmt, loc) {}
    ~if_stmt() override;
    void set_condition(expr* c) { condition = c; }
    auto get_condition() const { return condition; }
    void set_body(stmt* b) { body = b; }
    auto get_body() const { return body; }
    void set_else_body(stmt* b) { else_body = b; }
    auto get_else_body() const { return else_body; }
};

class for_stmt: public stmt {
private:
    std::string iter;
    range_expr* range = nullptr;
    block_stmt* body = nullptr;

public:
    for_stmt(const span& loc): stmt(ast::type::for_stmt, loc) {}
    ~for_stmt() override;
    void set_iter(const std::string& i) { iter = i; }
    const auto& get_iter() const { return iter; }
    void set_range(range_expr* r) { range = r; }
    auto get_range() const { return range; }
    void set_body(block_stmt* b) { body = b; }
    auto get_body() const { return body; }
};

class block_stmt: public stmt {
private:
    std::vector<stmt*> stmts;

public:
    block_stmt(const span& loc): stmt(ast::type::block_stmt, loc) {}
    ~block_stmt() override;
    void add_stmt(stmt* s) { stmts.push_back(s); }
    const auto& get_stmts() const { return stmts; }
};

}
