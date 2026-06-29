#include "dialect/dialect.hpp"

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
        cmp_eq_op,
        cmp_ne_op,
        cmp_lt_op,
        cmp_le_op,
        cmp_gt_op,
        cmp_ge_op,
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
        cast_op,
        constant_op,
        elements_op,
        stack_op,
        slice_op,
        yield_op,
        if_op,
        for_op
    >();
}

}
