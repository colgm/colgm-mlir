#pragma once

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

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

struct intrinsic_find_res {
    infer_func check;
    bool found = false;
};

class intrinsic_registry {
private:
    std::unordered_map<std::string, infer_func> intrinsics;

private:
    void regist(const char* name, infer_func fp) {
        intrinsics.emplace(name, fp);
    }

public:
    intrinsic_registry();
    intrinsic_find_res find(const std::string&) const;
};

type relu_infer(error&, call_expr*, type_storage&);
type abs_infer(error&, call_expr*, type_storage&);
type exp_infer(error&, call_expr*, type_storage&);
type log_infer(error&, call_expr*, type_storage&);
type sqrt_infer(error&, call_expr*, type_storage&);
type tanh_infer(error&, call_expr*, type_storage&);
type sigmoid_infer(error&, call_expr*, type_storage&);

}
