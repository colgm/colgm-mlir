#pragma once

#include "sema/type.hpp"
#include "sema/storage.hpp"

namespace colgm_mlir {

class context {
private:
    type_storage& ts;

public:
    context(type_storage& ts) : ts(ts) {}
};

}
