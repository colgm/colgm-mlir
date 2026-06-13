#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "ast/ast.hpp"
#include "ast/decl.hpp"
#include "ast/stmt.hpp"
#include "ast/expr.hpp"
#include "dialect/colgm/binary_op.hpp"
#include "dialect/colgm/unary_op.hpp"
#include "dialect/colgm/broadcast.hpp"
#include "dialect/colgm/transpose.hpp"
#include "dialect/colgm/reduce_sum.hpp"
#include "dialect/colgm/reshape.hpp"
#include "dialect/dialect.hpp"

namespace colgm_mlir {

class mlir_generator {
private:
    mlir::MLIRContext& ctx;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

private:
    mlir::Location to_loc(ast* node) {
        const auto& loc = node->get_location();
        return mlir::FileLineColLoc::get(
            builder.getStringAttr(loc.file),
            loc.begin_line,
            loc.begin_column + 1
        );
    }

    void generate_func(func_decl*);
    void generate_block(mlir::Block*, block_stmt*);
    void generate_stmt(mlir::Block*, stmt*);

public:
    mlir_generator(mlir::MLIRContext& c): ctx(c), builder(&c) {
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
