#pragma once

#include "utils/colgm.hpp"
#include "utils/span.hpp"

#include <string>

namespace colgm_mlir {

enum class tok: u32 {
    tk_null = 0,     // null token (default token type)
    tk_num,          // number literal
    tk_id,           // identifier
    tk_true,         // keyword true
    tk_false,        // keyword false
    tk_use,          // keyword use
    tk_for,          // loop keyword for
    tk_in,           // loop keyword in
    tk_var,          // keyword for definition
    tk_stct,         // keyword for struct
    tk_func,         // keyword for definition of function
    tk_ret,          // function keyword return
    tk_if,           // condition expression keyword if
    tk_else,         // condition expression keyword else
    tk_lcurve,       // (
    tk_rcurve,       // )
    tk_lbracket,     // [
    tk_rbracket,     // ]
    tk_lbrace,       // {
    tk_rbrace,       // }
    tk_semi,         // ;
    tk_comma,        // ,
    tk_dot,          // .
    tk_colon,        // :
    tk_double_colon, // ::
    tk_add,          // operator +
    tk_sub,          // operator -
    tk_mult,         // operator *
    tk_div,          // operator /
    tk_eq,           // operator =
    tk_cmpeq,        // operator ==
    tk_neq,          // operator !=
    tk_less,         // operator <
    tk_leq,          // operator <=
    tk_grt,          // operator >
    tk_geq,          // operator >=
    tk_arrow,        // operator ->
    tk_eof           // <eof> end of token list
};

struct token {
    span loc; // location
    tok type; // token type
    std::string str; // content
    token() = default;
    token(const token&) = default;
    token(const span& loc, tok type, const std::string& str):
        loc(loc), type(type), str(str) {}
};

}