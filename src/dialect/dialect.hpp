#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/BuiltinTypes.h>

namespace colgm_mlir {

class colgm_dialect: public mlir::Dialect {
public:
    explicit colgm_dialect(mlir::MLIRContext* ctx);
    static llvm::StringRef getDialectNamespace() { return "colgm"; }
};

}
