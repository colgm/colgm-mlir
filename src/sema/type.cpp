#include "sema/type.hpp"

namespace colgm_mlir {

size_t function_key_type_hash::operator()(const function_type::key_type& key) const {
    size_t hash = 0;
    for (const auto& t : std::get<0>(key)) {
        hash ^= std::hash<const void*>{}(t.get_impl()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    hash ^= std::hash<const void*>{}(std::get<1>(key).get_impl()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

size_t tensor_key_type_hash::operator()(const tensor_type::key_type& key) const {
    size_t hash = std::hash<const void*>{}(std::get<0>(key).get_impl());
    for (const auto& t : std::get<1>(key)) {
        hash ^= std::hash<i64>{}(t) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

}
