#pragma once

#include <string>
#include <unordered_map>

#include "sema/type.hpp"
#include "sema/storage.hpp"

namespace colgm_mlir {

class context {
private:
    type_storage& ts;
    std::unordered_map<std::string, type> functions;

public:
    context(type_storage& ts) : ts(ts) {}
    void regist_function(const std::string& name, type ft) {
        functions.emplace(name, ft);
    }
    void dump() const;
};

}
