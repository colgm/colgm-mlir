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

private:
    type resolve_type(type_def*);
    void resolve_stmt(stmt*);
    void resolve_var_decl(var_decl*);
    void resolve_assign_stmt(assign_stmt*);
    void resolve_return_stmt(return_stmt*);
    void resolve_if_stmt(if_stmt*);
    void resolve_for_stmt(for_stmt*);
    void resolve_block_stmt(block_stmt*);
    type resolve_expr(expr*);
    void resolve_func_decl(func_decl*);
    void resolve_func_block(func_decl*);

public:
    sema(error& e, type_storage& t): err(e), ts(t), ctx(t) {}
    const error& scan(root*);
    void dump() const;
};

}
