#pragma once

#include "utils/span.hpp"

namespace colgm_mlir {

class ast {
public:
    enum class type {
        type_def
    };

protected:
    type type_;
    span loc_;

public:
    ast(type t, span& loc): type_(t), loc_(loc) {}
    ~ast() = default;
};

}