#include <mlir/IR/OpImplementation.h>

#include "dialect/colgm/yield.hpp"
#include "dialect/colgm/if_op.hpp"
#include "dialect/colgm/for_op.hpp"

namespace colgm_mlir {

mlir::LogicalResult yield_op::verify() {
    auto parent = (*this)->getParentOp();
    if (!llvm::isa<if_op>(parent) && !llvm::isa<for_op>(parent)) {
        return emitOpError("yield must be inside if or for");
    }

    return mlir::success();
}

}
