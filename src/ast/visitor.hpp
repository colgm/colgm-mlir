#pragma once

#include "ast/ast.hpp"
#include "ast/decl.hpp"
#include "ast/stmt.hpp"
#include "ast/expr.hpp"

namespace colgm_mlir {

class visitor {
public:
    virtual bool visit_ast(ast*);
    virtual bool visit_decl(decl*);
    virtual bool visit_stmt(stmt*);
    virtual bool visit_expr(expr*);
    virtual bool visit_root(root*);

    virtual bool visit_int_literal(int_literal*);
    virtual bool visit_float_literal(float_literal*);
    virtual bool visit_bool_literal(bool_literal*);
    virtual bool visit_tensor(tensor*);
    virtual bool visit_identifier(identifier*);
    virtual bool visit_binary_expr(binary_expr*);
    virtual bool visit_unary_expr(unary_expr*);
    virtual bool visit_call_expr(call_expr*);
    virtual bool visit_index_access(index_access*);
    virtual bool visit_range_expr(range_expr*);

    virtual bool visit_var_decl(var_decl*);
    virtual bool visit_return_stmt(return_stmt*);
    virtual bool visit_yield_stmt(yield_stmt*);
    virtual bool visit_if_stmt(if_stmt*);
    virtual bool visit_for_stmt(for_stmt*);
    virtual bool visit_block_stmt(block_stmt*);

    virtual bool visit_func_decl(func_decl*);
    virtual bool visit_type_def(type_def*);
    virtual bool visit_param(param*);
};

}
