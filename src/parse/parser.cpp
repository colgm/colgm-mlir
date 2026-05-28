#include "parse/parser.hpp"

namespace colgm_mlir {

parser::~parser() {
    if (tree) {
        delete tree;
    }
}

bool parser::lookahead(tok t) const {
    return tokens[ptr].type == t;
}

const error& parser::scan() {
    tree = new root(tokens[0].loc);
    ptr = 0;
    while (!lookahead(tok::tk_eof)) {
        next();
    }

    return err;
}

}
