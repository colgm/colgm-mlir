#pragma once

#include "ast/ast.hpp"
#include "ast/visitor.hpp"

#include <vector>
#include <string>

namespace colgm_mlir {

class stmt: public ast {
public:
    stmt(ast_type t, const span& loc): ast(t, loc) {}
    ~stmt() override = default;
    void accept(visitor*) override;
};

class var_decl: public stmt {
private:
    std::string name;
    expr* init = nullptr;

public:
    var_decl(const span& loc): stmt(ast_type::var_decl, loc) {}
    ~var_decl() override;
    void accept(visitor*) override;
    void set_name(const std::string& n) { name = n; }
    const auto& get_name() const { return name; }
    void set_init(expr* i) { init = i; }
    auto get_init() const { return init; }
};

class return_stmt: public stmt {
private:
    expr* value = nullptr;

public:
    return_stmt(const span& loc): stmt(ast_type::return_stmt, loc) {}
    ~return_stmt() override;
    void accept(visitor*) override;
    void set_value(expr* v) { value = v; }
    auto get_value() const { return value; }
};

class yield_stmt: public stmt {
private:
    expr* value = nullptr;

public:
    yield_stmt(const span& loc): stmt(ast_type::yield_stmt, loc) {}
    ~yield_stmt() override;
    void accept(visitor*) override;
    void set_value(expr* v) { value = v; }
    auto get_value() const { return value; }
};

class expr_stmt: public stmt {
private:
    expr* inner = nullptr;

public:
    expr_stmt(const span& loc): stmt(ast_type::expr_stmt, loc) {}
    ~expr_stmt() override;
    void accept(visitor*) override;
    void set_inner(expr* e) { inner = e; }
    auto get_inner() const { return inner; }
};

class block_stmt: public stmt {
private:
    std::vector<stmt*> stmts;

public:
    block_stmt(const span& loc): stmt(ast_type::block_stmt, loc) {}
    ~block_stmt() override;
    void accept(visitor*) override;
    void add_stmt(stmt* s) { stmts.push_back(s); }
    const auto& get_stmts() const { return stmts; }
};

}
