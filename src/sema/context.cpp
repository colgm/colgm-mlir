#include "sema/context.hpp"

namespace colgm_mlir {

void context::dump() const {
    std::cout << "====== functions =====" << std::endl;
    for (const auto& [name, fi] : functions) {
        auto ft = type::as<function_type>(fi.func_type);
        std::cout << "func " << name << ft;
        std::cout << " [" << fi.loc << "]" << std::endl;
    }
    std::cout << "======================" << std::endl;
}

}
