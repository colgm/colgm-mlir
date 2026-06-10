#pragma once

#include <string>
#include <vector>
#include <functional>

#include "ast/ast.hpp"
#include "ast/expr.hpp"
#include "utils/colgm.hpp"
#include "report/report.hpp"
#include "sema/type.hpp"
#include "sema/storage.hpp"

namespace colgm_mlir {

typedef std::function<type(error&, call_expr*, type_storage&)> infer_func;

struct intrinsic {
    std::string name;
    infer_func infer;
};

type relu_infer(error&, call_expr*, type_storage&);

}
