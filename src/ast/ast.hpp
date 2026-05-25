#pragma once

#include "utils/span.hpp"

namespace colgm_mlir {

class ast {
public:
    enum class type {
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
        param
    };

protected:
    type type_;
    span loc_;

public:
    ast(type t, span& loc): type_(t), loc_(loc) {}
    virtual ~ast() = default;
};

class decl;
class expr;

}