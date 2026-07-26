#include "dialect/pass/scf_for_rank0_scalar.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/Pass.h>

namespace colgm_mlir {

// one-shot-bufferize requires yielded values inside scf.for to be buffer-equivalent
// to the corresponding iter_args. for rank-0 tensor iter_args (tensor<f64> etc.),
// no existing operation can perform an in-place update — arith.addf, linalg.generic,
// and tensor.from_elements all allocate new buffers. bufferization therefore cannot
// prove equivalence and fails with "yield operand not equivalent to iter bbArg".
//
// this pass works around the limitation by converting rank-0 tensor iter_args to
// scalar iter_args (f64 etc.), which bypass bufferization entirely. tensor↔scalar
// wrapping/unwrapping via from_elements/extract is inserted at loop boundaries,
// then canonicalized away, leaving pure scalar arithmetic in the loop body.
struct ScfForRank0Scalar
    : public mlir::PassWrapper<ScfForRank0Scalar,
                                mlir::OperationPass<mlir::func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScfForRank0Scalar)

    llvm::StringRef getArgument() const override {
        return "scf-for-rank0-to-scalar";
    }
    llvm::StringRef getDescription() const override {
        return "Convert scf.for rank-0 tensor iter_args to scalar iter_args";
    }

    void runOnOperation() override;
};

static bool hasRank0TensorIterArg(mlir::scf::ForOp forOp) {
    for (auto arg : forOp.getRegionIterArgs()) {
        auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(arg.getType());
        if (tensorTy && tensorTy.getRank() == 0) {
            return true;
        }
    }
    return false;
}

void rewriteForOp(mlir::scf::ForOp oldFor) {
    if (!hasRank0TensorIterArg(oldFor)) {
        return;
    }

    mlir::OpBuilder builder(oldFor);
    auto loc = oldFor.getLoc();
    unsigned numIterArgs = oldFor.getNumRegionIterArgs();

    auto oldInitArgs = oldFor.getInitArgs();
    auto oldIterArgs = oldFor.getRegionIterArgs();

    // track which iter_args are rank-0 tensors, and capture their original
    // types BEFORE modifying block args (ValueRange is live).
    llvm::SmallVector<bool> isRank0(numIterArgs, false);
    llvm::SmallVector<mlir::RankedTensorType> oldTensorTypes(numIterArgs);

    // step 1: extract scalars from rank-0 tensor init args
    llvm::SmallVector<mlir::Value> newInitArgs;
    for (unsigned i = 0; i < numIterArgs; ++i) {
        auto init = oldInitArgs[i];
        auto tensorTy =
            llvm::dyn_cast<mlir::RankedTensorType>(oldIterArgs[i].getType());
        if (tensorTy && tensorTy.getRank() == 0) {
            isRank0[i] = true;
            oldTensorTypes[i] = tensorTy;
            auto extract = mlir::tensor::ExtractOp::create(
                builder, loc, init, mlir::ValueRange{});
            newInitArgs.push_back(extract);
        } else {
            newInitArgs.push_back(init);
        }
    }

    // step 2: create new scf.for with scalar init args
    auto newFor = mlir::scf::ForOp::create(
        builder, loc, oldFor.getLowerBound(), oldFor.getUpperBound(),
        oldFor.getStep(), newInitArgs);

    // step 3: move the old body into the new for op
    auto& oldBody = oldFor.getBodyRegion();
    auto& newBody = newFor.getBodyRegion();

    // the new body has default blocks. Clear them and move old block in.
    newBody.getBlocks().clear();
    newBody.getBlocks().splice(newBody.end(), oldBody.getBlocks());

    // step 4: update block argument types for rank-0 tensor iter_args
    auto* entryBlock = &newBody.front();
    for (unsigned i = 0; i < numIterArgs; ++i) {
        if (isRank0[i]) {
            entryBlock->getArgument(i + 1).setType(
                oldTensorTypes[i].getElementType());
        }
    }

    // step 5: insert tensor.from_elements for each use of the now-scalar
    //         block arg, and replace uses with the wrapped tensor.
    for (unsigned i = 0; i < numIterArgs; ++i) {
        if (!isRank0[i]) continue;

        auto scalarArg = entryBlock->getArgument(i + 1);

        // collect all current uses
        llvm::SmallVector<mlir::OpOperand*> uses;
        for (auto& use : scalarArg.getUses()) {
            uses.push_back(&use);
        }

        // insert tensor.from_elements at the start of the block
        builder.setInsertionPointToStart(entryBlock);
        auto wrapped = mlir::tensor::FromElementsOp::create(
            builder, loc, oldTensorTypes[i], mlir::ValueRange{scalarArg});

        // replace all collected uses
        for (auto* use : uses) {
            use->set(wrapped);
        }
    }

    // step 6: before the scf.yield, extract scalars from rank-0 tensor
    //         yield operands
    auto yieldOp =
        llvm::cast<mlir::scf::YieldOp>(entryBlock->getTerminator());
    builder.setInsertionPoint(yieldOp);

    llvm::SmallVector<mlir::Value> newYieldOperands;
    for (auto [i, operand] : llvm::enumerate(yieldOp.getOperands())) {
        if (isRank0[i]) {
            auto extract = mlir::tensor::ExtractOp::create(
                builder, loc, operand, mlir::ValueRange{});
            newYieldOperands.push_back(extract);
        } else {
            newYieldOperands.push_back(operand);
        }
    }
    yieldOp->setOperands(newYieldOperands);

    // step 7: after the loop, wrap scalar results back to tensors
    builder.setInsertionPointAfter(newFor);

    llvm::SmallVector<mlir::Value> newResults;
    for (auto [i, result] : llvm::enumerate(newFor.getResults())) {
        if (isRank0[i]) {
            auto wrapped = mlir::tensor::FromElementsOp::create(
                builder, loc, oldTensorTypes[i], mlir::ValueRange{result});
            newResults.push_back(wrapped);
        } else {
            newResults.push_back(result);
        }
    }

    // replace old for results with new wrapped results
    for (auto [i, oldResult] : llvm::enumerate(oldFor.getResults())) {
        oldResult.replaceAllUsesWith(newResults[i]);
    }

    // erase the old (now-empty) for op
    oldFor.erase();
}

void ScfForRank0Scalar::runOnOperation() {
    llvm::SmallVector<mlir::scf::ForOp> forOps;
    getOperation()->walk(
        [&](mlir::scf::ForOp op) { forOps.push_back(op); });

    for (auto forOp : forOps) {
        rewriteForOp(forOp);
    }
}

std::unique_ptr<mlir::Pass> create_scf_for_rank0_scalar_pass() {
    return std::make_unique<ScfForRank0Scalar>();
}

} // namespace colgm_mlir
