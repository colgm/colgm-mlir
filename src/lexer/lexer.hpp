#pragma once

#include <cstring>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "utils/colgm.hpp"
#include "report/report.hpp"
#include "lexer/token.hpp"

namespace colgm_mlir {

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
        {"("       , tok::tk_lparen      },
        {")"       , tok::tk_rparen      },
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
