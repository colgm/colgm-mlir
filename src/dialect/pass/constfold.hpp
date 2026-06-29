#pragma once

#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <memory>

#include "dialect/colgm/elements.hpp"
#include "dialect/colgm/stack.hpp"
#include "dialect/colgm/constant_op.hpp"
#include "dialect/colgm/binary_op.hpp"
#include "dialect/colgm/unary_op.hpp"

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> create_colgm_const_fold_pass();

class colgm_const_fold_pass: public mlir::PassWrapper<colgm_const_fold_pass,
                                                      mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(colgm_const_fold_pass)

    llvm::StringRef getArgument() const override { return "colgm-const-fold"; }
    llvm::StringRef getDescription() const override {
        return "Fold IR with all-constant operands into colgm.constant / Do canonicalization";
    }
    void runOnOperation() override;
};

struct fold_neg : public mlir::OpRewritePattern<neg_op> {
    using mlir::OpRewritePattern<neg_op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(neg_op elem,
                                        mlir::PatternRewriter& rewriter) const override;
};

struct fold_elements : public mlir::OpRewritePattern<elements_op> {
    using mlir::OpRewritePattern<elements_op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(elements_op elem,
                                        mlir::PatternRewriter& rewriter) const override;
};

struct fold_stack : public mlir::OpRewritePattern<stack_op> {
    using mlir::OpRewritePattern<stack_op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(stack_op elem,
                                        mlir::PatternRewriter& rewriter) const override;
};

struct canonicalize_add : public mlir::OpRewritePattern<add_op> {
    using mlir::OpRewritePattern<add_op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(add_op elem,
                                        mlir::PatternRewriter& rewriter) const override;
};

struct canonicalize_sub : public mlir::OpRewritePattern<sub_op> {
    using mlir::OpRewritePattern<sub_op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(sub_op elem,
                                        mlir::PatternRewriter& rewriter) const override;
};

}

