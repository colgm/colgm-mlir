#include "sema/context.hpp"

namespace colgm_mlir {

void context::dump() const {
    std::cout << "====== functions =====" << std::endl;
    for (const auto& [name, type] : functions) {
        std::cout << "func " << name << type << std::endl;
    }
    std::cout << "======================" << std::endl;
}

}
