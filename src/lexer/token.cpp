#include "lexer/token.hpp"

namespace colgm_mlir {

const char* token::to_string(tok t) {
    switch (t) {
        case tok::tk_null:         return "<null>";
        case tok::tk_num:          return "<num>";
        case tok::tk_id:           return "<id>";
        case tok::tk_true:         return "true";
        case tok::tk_false:        return "false";
        case tok::tk_use:          return "use";
        case tok::tk_for:          return "for";
        case tok::tk_in:           return "in";
        case tok::tk_var:          return "var";
        case tok::tk_stct:         return "struct";
        case tok::tk_func:         return "func";
        case tok::tk_ret:          return "return";
        case tok::tk_if:           return "if";
        case tok::tk_else:         return "else";
        case tok::tk_lparen:       return "(";
        case tok::tk_rparen:       return ")";
        case tok::tk_lbracket:     return "[";
        case tok::tk_rbracket:     return "]";
        case tok::tk_lbrace:       return "{";
        case tok::tk_rbrace:       return "}";
        case tok::tk_semi:         return ";";
        case tok::tk_comma:        return ",";
        case tok::tk_dot:          return ".";
        case tok::tk_colon:        return ":";
        case tok::tk_double_colon: return "::";
        case tok::tk_add:          return "+";
        case tok::tk_sub:          return "-";
        case tok::tk_mult:         return "*";
        case tok::tk_div:          return "/";
        case tok::tk_eq:           return "=";
        case tok::tk_cmpeq:        return "==";
        case tok::tk_neq:          return "!=";
        case tok::tk_less:         return "<";
        case tok::tk_leq:          return "<=";
        case tok::tk_grt:          return ">";
        case tok::tk_geq:          return ">=";
        case tok::tk_arrow:        return "->";
        case tok::tk_eof:          return "<eof>";
    }

    return "";
}

std::string token::type_to_string() const {
    return std::string(token::to_string(type));
}

}
