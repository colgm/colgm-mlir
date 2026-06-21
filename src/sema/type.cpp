#include "sema/type.hpp"

namespace colgm_mlir {

std::ostream& operator<<(std::ostream& os, const type& t) {
    os << t.to_string();
    return os;
}

std::string function_type_impl::to_string() const {
    std::string res = "(";
    for (usize i = 0; i < args.size(); i++) {
        res += args[i].to_string();
        if (i != args.size() - 1) {
            res += ", ";
        }
    }
    res += ") -> " + ret.to_string();
    return res;
}

std::string tensor_type_impl::to_string() const {
    std::string res ="tensor<";
    for (const auto& t : shape) {
        res += std::to_string(t);
        res += " x ";
    }
    res += element_type.to_string();
    res += ">";
    return res;
}

usize function_key_type_hash::operator()(const function_type_impl::key_type& key) const {
    usize hash = 0;
    for (const auto& t : std::get<0>(key)) {
        hash ^= std::hash<const void*>{}(t.get_impl()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    hash ^= std::hash<const void*>{}(std::get<1>(key).get_impl()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

usize tensor_key_type_hash::operator()(const tensor_type_impl::key_type& key) const {
    usize hash = std::hash<const void*>{}(std::get<0>(key).get_impl());
    for (const auto& t : std::get<1>(key)) {
        hash ^= std::hash<i64>{}(t) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

}
