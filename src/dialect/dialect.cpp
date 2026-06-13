#include "dialect/dialect.hpp"

#include "dialect/colgm/binary_op.hpp"
#include "dialect/colgm/unary_op.hpp"
#include "dialect/colgm/reshape.hpp"
#include "dialect/colgm/transpose.hpp"
#include "dialect/colgm/broadcast.hpp"
#include "dialect/colgm/reduce_sum.hpp"
#include "dialect/colgm/constant_op.hpp"

namespace colgm_mlir {

colgm_dialect::colgm_dialect(mlir::MLIRContext* ctx):
    mlir::Dialect(
        getDialectNamespace(),
        ctx,
        mlir::TypeID::get<colgm_dialect>()
    ) {
    addOperations<
        add_op,
        sub_op,
        mul_op,
        div_op,
        matmul_op,
        relu_op,
        neg_op,
        abs_op,
        exp_op,
        log_op,
        sqrt_op,
        sigmoid_op,
        tanh_op,
        reshape_op,
        transpose_op,
        broadcast_op,
        reduce_sum,
        constant_op
    >();
}

}
