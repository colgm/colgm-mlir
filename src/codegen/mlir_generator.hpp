#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

#include "ast/ast.hpp"
#include "ast/decl.hpp"
#include "ast/stmt.hpp"
#include "ast/expr.hpp"
#include "dialect/colgm/binary_op.hpp"
#include "dialect/colgm/cast_op.hpp"
#include "dialect/colgm/cmp_op.hpp"
#include "dialect/colgm/constant_op.hpp"
#include "dialect/colgm/elements.hpp"
#include "dialect/colgm/if_op.hpp"
#include "dialect/colgm/for_op.hpp"
#include "dialect/colgm/slice.hpp"
#include "dialect/colgm/unary_op.hpp"
#include "dialect/colgm/broadcast.hpp"
#include "dialect/colgm/transpose.hpp"
#include "dialect/colgm/reduce_sum.hpp"
#include "dialect/colgm/reshape.hpp"
#include "dialect/colgm/yield.hpp"
#include "dialect/dialect.hpp"

namespace colgm_mlir {

struct var_stack {
    std::vector<std::unordered_map<std::string, mlir::Value>> var_stack;

    void add_scope() {
        var_stack.push_back(std::unordered_map<std::string, mlir::Value>());
    }

    void remove_scope() {
        var_stack.pop_back();
    }

    void add_var(std::string name, mlir::Value val) {
        var_stack.back()[name] = val;
    }

    mlir::Value get_var(const std::string& name) {
        for (auto it = var_stack.rbegin(); it != var_stack.rend(); ++it) {
            auto var = it->find(name);
            if (var != it->end()) {
                return var->second;
            }
        }
        assert(false && "variable not found, semantic check should catch this");
        return mlir::Value();
    }
};

class mlir_generator {
private:
    mlir::MLIRContext& ctx;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    var_stack vars;

private:
    mlir::Location to_loc(ast* node) {
        const auto& loc = node->get_location();
        return mlir::FileLineColLoc::get(
            builder.getStringAttr(loc.get_file()),
            loc.begin_line,
            loc.begin_column + 1
        );
    }

    mlir::Type convert_type(const type&);
    void flatten_tensor(std::vector<expr*>&, tensor*);

    void generate_func(func_decl*);
    void generate_block(mlir::Block*, block_stmt*);
    void generate_var_decl(mlir::Block*, var_decl*);
    void generate_return_stmt(mlir::Block*, return_stmt*);
    void generate_yield_stmt(mlir::Block*, yield_stmt*);
    void generate_expr_stmt(mlir::Block*, expr_stmt*);
    void generate_stmt(mlir::Block*, stmt*);
    mlir::Value generate_int_literal(int_literal*);
    mlir::Value generate_float_literal(float_literal*);
    mlir::Value generate_bool_literal(bool_literal*);
    mlir::Value generate_tensor(tensor*);
    mlir::Value generate_identifier(identifier*);
    mlir::Value generate_binary_expr(binary_expr*);
    mlir::Value generate_unary_expr(unary_expr*);
    mlir::Value generate_call_expr(call_expr*);
    mlir::Value generate_index_access(index_access*);
    mlir::Value generate_if_expr(if_expr*);
    mlir::Value generate_for_expr(for_expr*);
    mlir::Value generate_expr(expr*);

public:
    mlir_generator(mlir::MLIRContext& c): ctx(c), builder(&c) {
        ctx.getOrLoadDialect<colgm_dialect>();
        ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    }
    void generate(root*);
    void dump() {
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo();
        module.print(llvm::outs(), flags);
    }
};

}
