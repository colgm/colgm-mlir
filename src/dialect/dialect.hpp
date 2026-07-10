#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/BuiltinTypes.h>

#include "dialect/colgm/binary_op.hpp"
#include "dialect/colgm/unary_op.hpp"
#include "dialect/colgm/cmp_op.hpp"
#include "dialect/colgm/reshape.hpp"
#include "dialect/colgm/transpose.hpp"
#include "dialect/colgm/broadcast.hpp"
#include "dialect/colgm/reduce_sum.hpp"
#include "dialect/colgm/cast_op.hpp"
#include "dialect/colgm/constant_op.hpp"
#include "dialect/colgm/elements.hpp"
#include "dialect/colgm/stack.hpp"
#include "dialect/colgm/slice.hpp"
#include "dialect/colgm/yield.hpp"
#include "dialect/colgm/if_op.hpp"
#include "dialect/colgm/for_op.hpp"
#include "dialect/colgm/print_op.hpp"

namespace colgm_mlir {

class colgm_dialect: public mlir::Dialect {
public:
    explicit colgm_dialect(mlir::MLIRContext* ctx);
    static llvm::StringRef getDialectNamespace() { return "colgm"; }
};

}
