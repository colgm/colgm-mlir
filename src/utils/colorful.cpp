#include "utils/colorful.hpp"

#include <iostream>
#include <cstdio>

#ifdef _WIN32
    #include <io.h>
    #define isatty(fd) _isatty(fd)
    #define fileno(fd) _fileno(fd)
#else
    #include <unistd.h>
#endif

namespace colgm_mlir {

static bool stdout_is_tty() {
    static bool init = false;
    static bool tty = false;
    if (!init) {
        init = true;
        tty = isatty(fileno(stdout));
    }
    return tty;
}

static bool stderr_is_tty() {
    static bool init = false;
    static bool tty = false;
    if (!init) {
        init = true;
        tty = isatty(fileno(stderr));
    }
    return tty;
}

static bool enable_color(std::ostream& s) {
    if (s.rdbuf() == std::cout.rdbuf()) {
        return stdout_is_tty();
    }
    if (s.rdbuf() == std::cerr.rdbuf()) {
        return stderr_is_tty();
    }
    return false;
}

static std::ostream& colorful_core(std::ostream& s, const char* str) {
    if (!enable_color(s)) {
        return s;
    }
    s << str;
    return s;
}

std::ostream& back_white(std::ostream& s) {
    return colorful_core(s, "\033[7m");
}

std::ostream& red(std::ostream& s) {
    return colorful_core(s, "\033[91;1m");
}

std::ostream& green(std::ostream& s) {
    return colorful_core(s, "\033[92;1m");
}

std::ostream& orange(std::ostream& s) {
    return colorful_core(s, "\033[93;1m");
}

std::ostream& blue(std::ostream& s) {
    return colorful_core(s, "\033[94;1m");
}

std::ostream& purple(std::ostream& s) {
    return colorful_core(s, "\033[95;1m");
}

std::ostream& cyan(std::ostream& s) {
    return colorful_core(s, "\033[96;1m");
}

std::ostream& white(std::ostream& s) {
    return colorful_core(s, "\033[97;1m");
}

std::ostream& reset(std::ostream& s) {
    return colorful_core(s, "\033[0m");
}

}
