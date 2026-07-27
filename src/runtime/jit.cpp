#include "runtime/jit.hpp"

#include <dlfcn.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>

#include "runtime/print.hpp"
#include "dialect/pass/scf_for_rank0_scalar.hpp"

namespace colgm_mlir {

void run_jit(mlir::MLIRContext& context, colgm_mlir::codegen& gen) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::PassManager jit_pm(&context);
    jit_pm.addPass(mlir::createCanonicalizerPass());
    jit_pm.addNestedPass<mlir::func::FuncOp>(
        colgm_mlir::create_scf_for_rank0_scalar_pass());
    jit_pm.addPass(mlir::createConvertElementwiseToLinalgPass());
    {
      mlir::bufferization::OneShotBufferizePassOptions buf_opts;
      buf_opts.bufferizeFunctionBoundaries = true;
      jit_pm.addPass(mlir::bufferization::createOneShotBufferizePass(buf_opts));
    }
    jit_pm.addPass(mlir::createConvertLinalgToLoopsPass());
    jit_pm.addPass(mlir::createSCFToControlFlowPass());
    jit_pm.addPass(mlir::createConvertMathToLLVMPass());
    jit_pm.addPass(mlir::createArithToLLVMConversionPass());
    jit_pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    jit_pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    jit_pm.addPass(mlir::createConvertFuncToLLVMPass());
    jit_pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(jit_pm.run(gen.get_module()))) {
        llvm::errs() << "LLVM conversion failed\n";
        std::exit(-1);
    }

    // Make process symbols (including linked shared libraries like
    // libmlir_c_runner_utils) available to the JIT for resolving
    // runtime functions such as memrefCopy.
    //
    // First, load libmlir_c_runner_utils explicitly via dlopen.
    // The linker may optimize it out (--as-needed), so we must load
    // it at runtime to make memrefCopy available.
    dlopen("libmlir_c_runner_utils.so", RTLD_NOW | RTLD_GLOBAL);
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

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

}
