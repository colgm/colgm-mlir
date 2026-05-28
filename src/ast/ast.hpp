#pragma once

#include "utils/span.hpp"

#include <vector>

namespace colgm_mlir {

class visitor;

class ast {
public:
    enum class type {
        root,
        // expression
        int_literal,
        float_literal,
        bool_literal,
        identifier,
        binary_expr,
        unary_expr,
        call_expr,
        index_access,
        range_expr,
        // statement
        var_decl,
        assign_stmt,
        return_stmt,
        if_stmt,
        for_stmt,
        block_stmt,
        // declaration
        func_decl,
        type_def,
        param
    };

protected:
    type type_;
    span loc_;

public:
    ast(type t, const span& loc): type_(t), loc_(loc) {}
    virtual ~ast() = default;
    virtual void accept(visitor*);
};

class decl;
class stmt;
class expr;

class int_literal;
class float_literal;
class bool_literal;
class identifier;
class binary_expr;
class unary_expr;
class call_expr;
class index_access;
class range_expr;

class var_decl;
class assign_stmt;
class return_stmt;
class if_stmt;
class for_stmt;
class block_stmt;

class func_decl;
class type_def;
class param;

class root: public ast {
private:
    std::vector<func_decl*> funcs;

public:
    root(const span& loc): ast(ast::type::root, loc) {}
    ~root() override;
    void accept(visitor*) override;
    void add_func(func_decl* fd) { funcs.push_back(fd); }
    const auto& get_funcs() const { return funcs; }
};

}
