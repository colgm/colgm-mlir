#include "codegen/mlir_generator.hpp"

namespace colgm_mlir {

void mlir_generator::generate_func(func_decl* f) {
    builder.setInsertionPointToStart(module.getBody());
    auto func = mlir::func::FuncOp::create(
        builder, to_loc(f), f->get_name(),
        builder.getFunctionType({}, {}));

    auto* entry = func.addEntryBlock();
    generate_block(entry, f->get_body());

    builder.setInsertionPointToStart(entry);
    mlir::func::ReturnOp::create(builder, to_loc(f));
}

void mlir_generator::generate_block(mlir::Block* entry, block_stmt* b) {
    for (auto i : b->get_stmts()) {
        generate_stmt(entry, i);
    }
}

void mlir_generator::generate_stmt(mlir::Block* entry, stmt* s) {
    if (s->is(ast_type::block_stmt)) {
        generate_block(entry, static_cast<block_stmt*>(s));
    }
}

void mlir_generator::generate(root* r) {
    module = mlir::ModuleOp::create(builder, to_loc(r));

    for (auto f : r->get_funcs()) {
        generate_func(f);
    }
}

}
