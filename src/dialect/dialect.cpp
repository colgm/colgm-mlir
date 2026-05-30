#include "dialect/dialect.hpp"
#include "dialect/op.hpp"

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
        relu_op
    >();
}

}
