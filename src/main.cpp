#include "utils/colgm.hpp"
#include "lexer/lexer.hpp"

#include <vector>
#include <unordered_map>
#include <thread>
#include <cstdlib>

using colgm_mlir::u32;
using colgm_mlir::i32;

const u32 COMPILE_VIEW_TOKEN = 1;

std::ostream& help(std::ostream& out) {
    out
    << "\ncolgm-mlir <option>\n"
    << "option:\n"
    << "   -h,   --help           | get help.\n"
    << "   -v,   --version        | get version.\n"
    << "\ncolgm [option] <file>\n"
    << "option:\n"
    << "   -l,   --lex            | view analysed tokens.\n"
    << "file:\n"
    << "   <filename>             | input file.\n"
    << "\n";
    return out;
}

std::ostream& logo(std::ostream& out) {
    out
    << "             .__                                    .__  .__        \n"
    << "  ____  ____ |  |    ____   _____             _____ |  | |__|______ \n"
    << "_/ ___\\/  _ \\|  |   / ___\\ /     \\   ______  /     \\|  | |  \\_  __ \\\n"
    << "\\  \\__(  <_> )  |__/ /_/  >  Y Y  \\ /_____/ |  Y Y  \\  |_|  ||  | \\/\n"
    << " \\___  >____/|____/\\___  /|__|_|  /         |__|_|  /____/__||__|   \n"
    << "     \\/           /_____/       \\/                \\/                \n\n"
    << "version : " << __colgm_ver__
    << " " << colgm_mlir::get_platform() << " " << colgm_mlir::get_arch()
    << " (" << __DATE__ << " " << __TIME__ << ")\n"
    << "repo    : https://github.com/colgm/colgm-mlir\n"
    << "license : Apache 2.0\n"
    << "\n"
    << "input <colgm -h> to get help.\n\n";
    return out;
}

std::ostream& version(std::ostream& out) {
    out << "colgm-mlir compiler version " << __colgm_ver__;
    out << " " << colgm_mlir::get_platform() << " " << colgm_mlir::get_arch();
    out << " (" << __DATE__ << " " << __TIME__ << ")\n";
    return out;
}

[[noreturn]]
void err() {
    std::cerr
    << "invalid argument(s).\n"
    << "use <colgm-mlir -h> to get help.\n";
    std::exit(1);
}

void execute(const std::string& input_file,
             const std::string& output_file,
             const u32 cmd = 0) {
    // main components of compiler
    colgm_mlir::error err;
    colgm_mlir::lexer lexer(err);

    // lexer scans file to get tokens
    lexer.scan(input_file).chkerr();
    if (cmd & COMPILE_VIEW_TOKEN) {
        for (const auto& token : lexer.result()) {
            std::cout << token.loc << ": " << token.str << "\n";
        }
    }
}

i32 main(i32 argc, const char* argv[]) {
    // output version info
    if (argc <= 1) {
        std::clog << logo;
        return 0;
    }

    // run directly or show help
    if (argc == 2) {
        std::string s(argv[1]);
        if (s == "-h" || s == "--help") {
            std::clog << help;
        } else if (s == "-v" || s == "--version") {
            std::clog << version;
        } else if (s[0] != '-') {
            execute(s, "a.out.ll");
        } else {
            err();
        }
        return 0;
    }

    // execute with arguments
    const std::unordered_map<std::string, u32> cmdlst = {
        {"--lex", COMPILE_VIEW_TOKEN},
        {"-l", COMPILE_VIEW_TOKEN},
    };
    u32 cmd = 0;
    std::string input_file = "";
    std::string output_file = "a.out.ll";

    std::vector<std::string> args;
    for (i32 i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }

    for (i32 i = 1; i < argc; ++i) {
        if (args[i] == "-h" || args[i] == "--help") {
            std::clog << help;
            return 0;
        } else if (cmdlst.count(args[i])) {
            cmd |= cmdlst.at(args[i]);
        } else if (args[i] == "-o" || args[i] == "--output") {
            if (i + 1 < argc) {
                output_file = args[i + 1];
                ++i;
            } else {
                err();
            }
        } else if (args[i] == "--arch") {
            if (i + 1 < argc) {
                colgm_mlir::target_info::singleton()->set_arch(args[i + 1]);
                ++i;
            } else {
                err();
            }
        } else if (args[i] == "--platform") {
            if (i + 1 < argc) {
                colgm_mlir::target_info::singleton()->set_platform(args[i + 1]);
                ++i;
            } else {
                err();
            }
        } else if (!input_file.length()) {
            input_file = args[i];
        } else {
            err();
        }
    }

    // input file must be specified
    if (!input_file.length()) {
        err();
    }

    execute(input_file, output_file, cmd);
    return 0;
}