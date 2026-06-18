#pragma once

#include <string>
#include <unordered_set>

namespace colgm_mlir {

class source_manager {
private:
    std::unordered_set<std::string> files;

public:
    const std::string* intern(const std::string& filename) {
        return &(*files.insert(filename).first);
    }
};

}
