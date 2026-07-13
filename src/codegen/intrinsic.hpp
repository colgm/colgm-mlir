#pragma once

#include <string>
#include <functional>
#include <unordered_map>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Location.h>

#include "dialect/colgm/unary_op.hpp"
#include "dialect/dialect.hpp"

namespace colgm_mlir {

typedef std::function<mlir::Value(mlir::OpBuilder&,
                                  mlir::Location,
                                  llvm::SmallVector<mlir::Value>&)> gen_func;

struct intrinsic_gen_find_res {
    gen_func generate;
    bool found = false;
};

class intrinsic_generator_registry {
private:
    std::unordered_map<std::string, gen_func> intrinsic_generators;

private:
    void regist(const char* name, gen_func fp) {
        intrinsic_generators.emplace(name, fp);
    }

public:
    intrinsic_generator_registry();
    intrinsic_gen_find_res find(const std::string&) const;
};

mlir::Value relu_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);
mlir::Value abs_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);
mlir::Value exp_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);
mlir::Value log_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);
mlir::Value sqrt_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);
mlir::Value tanh_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);
mlir::Value sigmoid_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);
mlir::Value print_gen(mlir::OpBuilder&, mlir::Location, llvm::SmallVector<mlir::Value>&);

}
