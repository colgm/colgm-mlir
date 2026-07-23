#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <cstdlib>

#include <mlir/Support/LLVM.h>
#include <mlir/Pass/PassManager.h>

#include "utils/colgm.hpp"
#include "utils/source_manager.hpp"
#include "lexer/lexer.hpp"
#include "ast/dumper.hpp"
#include "parse/parser.hpp"
#include "sema/storage.hpp"
#include "sema/sema.hpp"

#include "dialect/dialect.hpp"
#include "codegen/generator.hpp"
#include "dialect/pass/constfold.hpp"
#include "dialect/pass/lowering.hpp"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>

#include "runtime/print.hpp"

using colgm_mlir::u32;
using colgm_mlir::i32;

const u32 COMPILE_VIEW_TOKEN = 1;
const u32 COMPILE_VIEW_AST = 1 << 1;
const u32 COMPILE_VIEW_SEMA = 1 << 2;
const u32 COMPILE_VIEW_SEMA_AST = 1 << 3;
const u32 COMPILE_VIEW_MLIR = 1 << 4;

struct pass_options {
    const std::unordered_set<std::string> available = {
        "--const-fold",
        "--lowering",
        "--jit"
    };
    std::unordered_set<std::string> options;

    bool is_available_pass(const std::string& name) const {
        return available.count(name) != 0;
    }
    void add_pass(const std::string& name) {
        if (is_available_pass(name)) {
            options.insert(name);
        }
    }
    bool enable_const_fold() const {
        return options.count("--const-fold") != 0;
    }
    bool enable_lowering() const {
        return options.count("--lowering") != 0;
    }
    bool enable_jit() const {
        return options.count("--jit") != 0;
    }
};

std::ostream& help(std::ostream& out) {
    out
    << "\ncolgm-mlir <option>\n"
    << "option:\n"
    << "   -h,   --help           | get help.\n"
    << "   -v,   --version        | get version.\n"
    << "\ncolgm [option] <file>\n"
    << "option:\n"
    << "   -l,   --lex            | view analysed tokens.\n"
    << "   -a,   --ast            | view analysed ast.\n"
    << "   -s,   --sema           | view analysed semantic context.\n"
    << "   -sa,  --sema-ast       | view analysed ast with semantic result.\n"
    << "   -m,   --mlir           | view generated mlir.\n"
    << "         --const-fold     | enable const fold/canonicalization.\n"
    << "         --lowering       | enable colgm dialect lowering.\n"
    << "         --jit            | JIT compile and execute.\n"
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
    << " (" << __DATE__ << " " << __TIME__ << ")\n"
    << "arch    : " << colgm_mlir::get_platform() << " " << colgm_mlir::get_arch() << "\n"
    << "mlir    : " << LLVM_VERSION_STRING << "\n"
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
    std::cerr << "invalid argument(s).\n"
              << "use <colgm-mlir -h> to get help.\n";
    std::exit(1);
}

void run_jit(mlir::MLIRContext& context, colgm_mlir::codegen& gen) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::PassManager jit_pm(&context);
    jit_pm.addPass(mlir::createCanonicalizerPass());
    jit_pm.addPass(mlir::createConvertElementwiseToLinalgPass());
    jit_pm.addPass(mlir::bufferization::createOneShotBufferizePass());
    jit_pm.addPass(mlir::createConvertLinalgToLoopsPass());
    jit_pm.addPass(mlir::createSCFToControlFlowPass());
    jit_pm.addPass(mlir::createConvertMathToLLVMPass());
    jit_pm.addPass(mlir::createArithToLLVMConversionPass());
    jit_pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    jit_pm.addPass(mlir::createConvertFuncToLLVMPass());
    jit_pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    jit_pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(jit_pm.run(gen.get_module()))) {
        llvm::errs() << "LLVM conversion failed\n";
        std::exit(-1);
    }

    auto maybeEngine = mlir::ExecutionEngine::create(gen.get_module());
    if (!maybeEngine) {
        llvm::errs() << "Failed to create ExecutionEngine: "
                        << maybeEngine.takeError() << "\n";
        std::exit(-1);
    }
    auto& engine = *maybeEngine;

    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap map;
        map[interner("__colgm_print_f64")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(&__colgm_print_f64);
        map[interner("__colgm_print_f32")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(&__colgm_print_f32);
        map[interner("__colgm_print_i64")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(&__colgm_print_i64);
        map[interner("__colgm_print_i32")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(&__colgm_print_i32);
        map[interner("__colgm_print_i1")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(&__colgm_print_i1);
        map[interner("__colgm_print_open_bracket")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(
                &__colgm_print_open_bracket);
        map[interner("__colgm_print_close_bracket")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(
                &__colgm_print_close_bracket);
        map[interner("__colgm_print_comma")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(&__colgm_print_comma);
        map[interner("__colgm_print_newline")] =
            llvm::orc::ExecutorSymbolDef::fromPtr(&__colgm_print_newline);
        return map;
    });

    auto main_fn = engine->lookup("main");
    if (!main_fn) {
        llvm::errs() << "Failed to lookup 'main': "
                        << main_fn.takeError() << "\n";
        std::exit(-1);
    }
    auto* main_ptr = reinterpret_cast<void (*)()>(*main_fn);
    main_ptr();
    return;
}

void execute(const std::string& input_file,
             const std::string& output_file,
             const pass_options& po,
             const u32 cmd = 0) {
    // main components of compiler
    colgm_mlir::error err;
    colgm_mlir::source_manager sm;
    colgm_mlir::lexer lexer(err);

    // lexer scans file to get tokens
    lexer.scan(sm.intern(input_file)).chkerr();
    if (cmd & COMPILE_VIEW_TOKEN) {
        for (const auto& token : lexer.result()) {
            std::cout << token.loc << ": " << token.str << "\n";
        }
        return;
    }

    colgm_mlir::parser parser(lexer.result(), err);
    parser.scan().chkerr();
    if (cmd & COMPILE_VIEW_AST) {
        colgm_mlir::dumper::dump(parser.get_tree());
        return;
    }

    colgm_mlir::type_storage ts;
    colgm_mlir::sema sema(err, ts);
    sema.scan(parser.get_tree()).chkerr();
    if (cmd & COMPILE_VIEW_SEMA) {
        sema.dump();
        return;
    } else if (cmd & COMPILE_VIEW_SEMA_AST) {
        colgm_mlir::dumper::dump(parser.get_tree());
        return;
    }

    mlir::MLIRContext context;
    {
        mlir::DialectRegistry registry;
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerConvertMathToLLVMInterface(registry);
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
        context.appendDialectRegistry(registry);
    }
    colgm_mlir::codegen gen(context);

    mlir::PassManager pm(&context);
    bool jit_mode = po.enable_jit();
    if (po.enable_const_fold() || jit_mode) {
        pm.addNestedPass<mlir::func::FuncOp>(colgm_mlir::create_colgm_const_fold_pass());
    }
    if (po.enable_lowering() || jit_mode) {
        pm.addPass(colgm_mlir::create_colgm_lowering_pass());
    }

    gen.generate(parser.get_tree());
    if (mlir::failed(gen.get_module().verify())) {
        gen.dump();
        std::exit(-1);
    }
    if (mlir::failed(pm.run(gen.get_module()))) {
        std::exit(-1);
    }

    if (cmd & COMPILE_VIEW_MLIR) {
        gen.dump();
        return;
    }

    if (po.enable_jit()) {
        run_jit(context, gen);
        return;
    }
}

i32 main(i32 argc, const char* argv[]) {
    // output version info
    if (argc <= 1) {
        std::clog << logo;
        return 0;
    }

    pass_options po;

    // run directly or show help
    if (argc == 2) {
        std::string s(argv[1]);
        if (s == "-h" || s == "--help") {
            std::clog << help;
        } else if (s == "-v" || s == "--version") {
            std::clog << version;
        } else if (s[0] != '-') {
            execute(s, "a.out.ll", po);
        } else {
            err();
        }
        return 0;
    }

    // execute with arguments
    const std::unordered_map<std::string, u32> cmdlst = {
        { "--lex", COMPILE_VIEW_TOKEN },
        { "-l",    COMPILE_VIEW_TOKEN },
        { "--ast", COMPILE_VIEW_AST },
        { "-a",    COMPILE_VIEW_AST },
        { "--sema", COMPILE_VIEW_SEMA },
        { "-s",    COMPILE_VIEW_SEMA },
        { "--sema-ast", COMPILE_VIEW_SEMA_AST },
        { "-sa",    COMPILE_VIEW_SEMA_AST },
        { "--mlir", COMPILE_VIEW_MLIR },
        { "-m",    COMPILE_VIEW_MLIR }
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
        } else if (po.is_available_pass(args[i])) {
            po.add_pass(args[i]);
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

    execute(input_file, output_file, po, cmd);
    return 0;
}