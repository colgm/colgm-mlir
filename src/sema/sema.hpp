#pragma once

#include "ast/ast.hpp"
#include "ast/decl.hpp"
#include "ast/expr.hpp"
#include "ast/stmt.hpp"
#include "report/report.hpp"
#include "sema/type.hpp"
#include "sema/storage.hpp"
#include "sema/context.hpp"

namespace colgm_mlir {

class sema {
private:
    error& err;
    type_storage& ts;
    context ctx;
    type func_ret_type;

private:
    bool block_contains_return(block_stmt*);
    yield_stmt* find_yield(block_stmt*);

private:
    type resolve_type(type_def*);
    void resolve_stmt(stmt*, bool);
    void resolve_var_decl(var_decl*);
    void resolve_return_stmt(return_stmt*);
    void resolve_yield_stmt(yield_stmt*);
    void resolve_expr_stmt(expr_stmt*);
    void resolve_block_stmt(block_stmt*);
    type resolve_int_literal(int_literal*);
    type resolve_float_literal(float_literal*);
    type resolve_bool_literal(bool_literal*);
    type resolve_tensor(tensor*);
    type resolve_identifier(identifier*);
    type resolve_binary_expr(binary_expr*);
    type resolve_unary_expr(unary_expr*);
    type resolve_call_expr(call_expr*);
    type resolve_index_access(index_access*);
    type resolve_range_expr(range_expr*);
    type resolve_if_expr(if_expr*);
    type resolve_for_expr(for_expr*);
    type resolve_expr(expr*);
    void resolve_func_decl(func_decl*);
    void resolve_func_block(func_decl*);

public:
    sema(error& e, type_storage& t):
        err(e), ts(t), ctx(t), func_ret_type(ts.get_void_type()) {}
    const error& scan(root*);
    void dump() const;
};

}
