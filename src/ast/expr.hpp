#pragma once

#include "utils/type.hpp"
#include "ast/ast.hpp"

#include <string>

namespace colgm_mlir {

class expr: public ast {
public:
    expr(ast::type t, span& loc): ast(t, loc) {}
    ~expr() override = default;
};

class int_literal: public expr {
private:
    i64 literal;

public:
    int_literal(span& loc): expr(ast::type::int_literal, loc) {}
    void set_literal(i64 lit) { literal = lit; }
    auto get_literal() const { return literal; }
};

class float_literal: public expr {
private:
    f64 literal;

public:
    float_literal(span& loc): expr(ast::type::float_literal, loc) {}
    void set_literal(f64 lit) { literal = lit; }
    auto get_literal() const { return literal; }
};

class bool_literal: public expr {
private:
    bool flag;

public:
    bool_literal(span& loc): expr(ast::type::bool_literal, loc) {}
    void set_flag(bool f) { flag = f; }
    auto get_flag() const { return flag; }
};

class identifier: public expr {
private:
    std::string name;

public:
    identifier(span& loc): expr(ast::type::identifier, loc) {}
    void set_name(const std::string& n) { name = n; }
    const auto& get_name() const { return name; }
};

}
