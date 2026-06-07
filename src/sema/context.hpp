#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "utils/span.hpp"
#include "sema/type.hpp"
#include "sema/storage.hpp"

namespace colgm_mlir {

struct func_info {
    std::vector<std::string> args;
    std::unordered_map<std::string, type> args_types;
    type func_type;
    span loc;

    bool is_generic_function = false;

    func_info(const span& l, type ft): func_type(ft), loc(l) {
        for (const auto& [_, t] : args_types) {
            if (type::isa<unknown_type>(t)) {
                is_generic_function = true;
                break;
            }
        }
    }
    void add_arg(const std::string& name, type t) {
        args.push_back(name);
        args_types.emplace(name, t);
    }
};

class context {
private:
    type_storage& ts;
    std::unordered_map<std::string, func_info> functions;

public:
    context(type_storage& ts) : ts(ts) {}
    void regist_function(const std::string& name, func_info&& fi) {
        functions.emplace(name, fi);
    }
    const auto& get_functions() const { return functions; }
    void dump() const;
};

}
