#pragma once

#include <cstring>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "colgm.hpp"
#include "report.hpp"

namespace colgm_mlir {

enum class tok:u32 {
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

class lexer {
private:
    u32 line;
    u32 column;
    usize ptr;
    std::string filename;
    std::string res;

    error& err;
    u64 invalid_char;
    std::vector<token> toks;

    const std::unordered_map<std::string, tok> type_table = {
        {"use"     , tok::tk_use         },
        {"true"    , tok::tk_true        },
        {"false"   , tok::tk_false       },
        {"for"     , tok::tk_for         },
        {"in"      , tok::tk_in          },
        {"var"     , tok::tk_var         },
        {"struct"  , tok::tk_stct        },
        {"func"    , tok::tk_func        },
        {"return"  , tok::tk_ret         },
        {"if"      , tok::tk_if          },
        {"else"    , tok::tk_else        },
        {"("       , tok::tk_lcurve      },
        {")"       , tok::tk_rcurve      },
        {"["       , tok::tk_lbracket    },
        {"]"       , tok::tk_rbracket    },
        {"{"       , tok::tk_lbrace      },
        {"}"       , tok::tk_rbrace      },
        {";"       , tok::tk_semi        },
        {","       , tok::tk_comma       },
        {"."       , tok::tk_dot         },
        {":"       , tok::tk_colon       },
        {"::"      , tok::tk_double_colon},
        {"+"       , tok::tk_add         },
        {"-"       , tok::tk_sub         },
        {"*"       , tok::tk_mult        },
        {"/"       , tok::tk_div         },
        {"="       , tok::tk_eq          },
        {"=="      , tok::tk_cmpeq       },
        {"!="      , tok::tk_neq         },
        {"<"       , tok::tk_less        },
        {"<="      , tok::tk_leq         },
        {">"       , tok::tk_grt         },
        {">="      , tok::tk_geq         },
        {"->"      , tok::tk_arrow       }
    };

    tok get_type(const std::string&);
    bool skip(char);
    bool is_id(char);
    bool is_dec(char);
    bool is_single_opr(char);
    bool is_calc_opr(char);
    bool is_arrow(char);
    bool is_wide_arrow(char);
    bool is_note();
    bool is_multi_line_comment();

    void skip_note();
    void skip_multi_line_comment();
    void err_char();

    void open(const std::string&);
    std::string utf8_gen();
    token id_gen();
    token num_gen();
    token arrow_gen();
    token single_opr();
    token dots();
    token colons();
    token calc_opr();

public:
    lexer(error& e):
        line(1), column(0), ptr(0),
        filename(""), res(""), err(e), invalid_char(0) {}
    const error& scan(const std::string&);
    const auto& result() const { return toks; }
};

}
