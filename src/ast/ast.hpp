#pragma once

#include <vector>

#include "utils/span.hpp"
#include "sema/type.hpp"

namespace colgm_mlir {

enum class ast_type {
    root,
    // expression
    int_literal,
    float_literal,
    bool_literal,
    tensor,
    identifier,
    binary_expr,
    unary_expr,
    call_expr,
    index_access,
    range_expr,
    // statement
    var_decl,
    return_stmt,
    if_stmt,
    for_stmt,
    block_stmt,
    // declaration
    func_decl,
    type_def,
    param
};

class visitor;

class ast {
protected:
    ast_type ast_type_;
    span loc_;
    type resolve_type_;

public:
    ast(ast_type t, const span& loc):
        ast_type_(t), loc_(loc), resolve_type_(nullptr) {}
    virtual ~ast() = default;
    const auto& get_location() const { return loc_; }
    void set_resolved(const type& t) { resolve_type_ = t; }
    const auto get_resolved() const { return resolve_type_; }
    auto get_ast_type() const { return ast_type_; }
    void update_location(const span& end) {
        loc_.end_line = end.end_line;
        loc_.end_column = end.end_column;
    }
    virtual void accept(visitor*);
};

class decl;
class stmt;
class expr;

class int_literal;
class float_literal;
class bool_literal;
class tensor;
class identifier;
class binary_expr;
class unary_expr;
class call_expr;
class index_access;
class range_expr;

class var_decl;
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
    root(const span& loc): ast(ast_type::root, loc) {}
    ~root() override;
    void accept(visitor*) override;
    void add_func(func_decl* fd) { funcs.push_back(fd); }
    const auto& get_funcs() const { return funcs; }
};

}
