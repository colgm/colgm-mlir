#pragma once

#include "utils/type.hpp"
#include "ast/ast.hpp"
#include "ast/visitor.hpp"

#include <string>
#include <vector>
#include <optional>

namespace colgm_mlir {

class expr: public ast {
public:
    expr(ast_type t, const span& loc): ast(t, loc) {}
    ~expr() override = default;
    void accept(visitor*) override;
};

class int_literal: public expr {
private:
    i64 literal;

public:
    int_literal(const span& loc): expr(ast_type::int_literal, loc) {}
    ~int_literal() override = default;
    void accept(visitor*) override;
    void set_literal(i64 lit) { literal = lit; }
    auto get_literal() const { return literal; }
};

class float_literal: public expr {
private:
    f64 literal;

public:
    float_literal(const span& loc): expr(ast_type::float_literal, loc) {}
    ~float_literal() override = default;
    void accept(visitor*) override;
    void set_literal(f64 lit) { literal = lit; }
    auto get_literal() const { return literal; }
};

class bool_literal: public expr {
private:
    bool flag;

public:
    bool_literal(const span& loc): expr(ast_type::bool_literal, loc) {}
    ~bool_literal() override = default;
    void accept(visitor*) override;
    void set_flag(bool f) { flag = f; }
    auto get_flag() const { return flag; }
};

class tensor: public expr {
private:
    std::vector<expr*> values;

public:
    tensor(const span& loc): expr(ast_type::tensor, loc) {}
    ~tensor() override;
    void accept(visitor*) override;
    void add_value(expr* v) { values.push_back(v); }
    const auto& get_values() const { return values; }
};

class identifier: public expr {
private:
    std::string name;

public:
    identifier(const span& loc): expr(ast_type::identifier, loc) {}
    ~identifier() override = default;
    void accept(visitor*) override;
    void set_name(const std::string& n) { name = n; }
    const auto& get_name() const { return name; }
};

class binary_expr: public expr {
public:
    enum class op {
        add,
        sub,
        mul,
        div
    };

private:
    op op_type;
    expr* lhs = nullptr;
    expr* rhs = nullptr;

public:
    binary_expr(const span& loc): expr(ast_type::binary_expr, loc) {}
    ~binary_expr() override;
    void accept(visitor*) override;
    void set_op_type(op o) { op_type = o; }
    void set_lhs(expr* l) { lhs = l; }
    void set_rhs(expr* r) { rhs = r; }
    auto get_op_type() const { return op_type; }
    auto get_lhs() const { return lhs; }
    auto get_rhs() const { return rhs; }
};

class unary_expr: public expr {
public:
    enum class op {
        add,
        sub
    };

private:
    op op_type;
    expr* operand = nullptr;

public:
    unary_expr(const span& loc): expr(ast_type::unary_expr, loc) {}
    ~unary_expr() override;
    void accept(visitor*) override;
    void set_op_type(op o) { op_type = o; }
    void set_operand(expr* o) { operand = o; }
    auto get_op_type() const { return op_type; }
    auto get_operand() const { return operand; }
};

class call_expr: public expr {
private:
    expr* callee = nullptr;
    std::vector<expr*> args;

public:
    call_expr(const span& loc): expr(ast_type::call_expr, loc) {}
    ~call_expr() override;
    void accept(visitor*) override;
    void set_callee(expr* c) { callee = c; }
    void add_arg(expr* a) {
        args.push_back(a);
    }
    auto get_callee() const { return callee; }
    const auto& get_args() const { return args; }
};

class index_access: public expr {
private:
    expr* target = nullptr;
    expr* index = nullptr;

public:
    index_access(const span& loc): expr(ast_type::index_access, loc) {}
    ~index_access() override;
    void accept(visitor*) override;
    void set_target(expr* t) { target = t; }
    void set_index(expr* i) { index = i; }
    auto get_target() const { return target; }
    auto get_index() const { return index; }
};

class range_expr: public expr {
private:
    expr* start = nullptr;
    expr* end = nullptr;

public:
    range_expr(const span& loc): expr(ast_type::range_expr, loc) {}
    ~range_expr() override;
    void accept(visitor*) override;
    void set_start(expr* s) { start = s; }
    void set_end(expr* e) { end = e; }
    auto get_start() const { return start; }
    auto get_end() const { return end; }
};

}
