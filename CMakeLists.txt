cmake_minimum_required(VERSION 3.20)
project(colgm-mlir
    DESCRIPTION "Experimental MLIR-based compiler"
    LANGUAGES CXX C
)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- platform detection ----
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# ---- MLIR / LLVM (optional, for future use) ----
set(LLVM_DIR "" CACHE PATH "Path to LLVM cmake directory")
set(MLIR_DIR "" CACHE PATH "Path to MLIR cmake directory")

option(ENABLE_MLIR "Enable MLIR support" OFF)

if(ENABLE_MLIR)
    # zstd is an optional LLVM dependency for compressed debug-info / bitcode.
    # Our pipeline (text .mlir → IR → JIT) never touches compressed data, so an
    # empty stub target is safe.  Install libzstd-dev if you later need to read
    # compressed .bc.zst files.
    if(NOT TARGET zstd::libzstd_static)
      add_library(zstd::libzstd_static INTERFACE IMPORTED)
    endif()

    # try to auto-discover from the local bin/ directory
    if(NOT LLVM_DIR OR NOT MLIR_DIR)
        file(GLOB LLVM_DIRS LIST_DIRECTORIES true
            "${CMAKE_SOURCE_DIR}/bin/LLVM-*")
        if(LLVM_DIRS)
            list(GET LLVM_DIRS 0 LLVM_PREFIX)
            set(LLVM_DIR "${LLVM_PREFIX}/lib/cmake/llvm")
            set(MLIR_DIR "${LLVM_PREFIX}/lib/cmake/mlir")
        endif()
    endif()

    if(LLVM_DIR AND MLIR_DIR)
        find_package(LLVM REQUIRED CONFIG PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
        find_package(MLIR REQUIRED CONFIG PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
        message(STATUS "MLIR support enabled (LLVM ${LLVM_PACKAGE_VERSION})")
    else()
        message(FATAL_ERROR "ENABLE_MLIR=ON but LLVM_DIR or MLIR_DIR not found. "
                            "Run scripts/download_llvm.py first.")
    endif()
endif()

# ---- source files ----
set(COLGM_LIB_SOURCES
    src/ast/ast.cpp
    src/ast/decl.cpp
    src/ast/expr.cpp
    src/ast/stmt.cpp
    src/ast/visitor.cpp
    src/ast/dumper.cpp
    src/codegen/mlir_generator.cpp
    src/dialect/colgm/binary_op.cpp
    src/dialect/colgm/cmp_op.cpp
    src/dialect/colgm/broadcast.cpp
    src/dialect/colgm/cast_op.cpp
    src/dialect/colgm/constant_op.cpp
    src/dialect/colgm/elements.cpp
    src/dialect/colgm/for_op.cpp
    src/dialect/colgm/if_op.cpp
    src/dialect/colgm/reduce_sum.cpp
    src/dialect/colgm/reshape.cpp
    src/dialect/colgm/slice.cpp
    src/dialect/colgm/transpose.cpp
    src/dialect/colgm/unary_op.cpp
    src/dialect/colgm/yield.cpp
    src/dialect/pass/constfold.cpp
    src/dialect/dialect.cpp
    src/lexer/lexer.cpp
    src/lexer/token.cpp
    src/parse/parser.cpp
    src/report/report.cpp
    src/sema/context.cpp
    src/sema/intrinsic.cpp
    src/sema/sema.cpp
    src/sema/storage.cpp
    src/sema/type.cpp
    src/utils/colorful.cpp
    src/utils/misc.cpp
    src/utils/span.cpp
)
set(COLGM_SOURCES
    ${COLGM_LIB_SOURCES}
    src/main.cpp
)

# ---- colgm mlir core library ----
add_library(colgm_mlir_core STATIC
    ${COLGM_LIB_SOURCES}
)

target_include_directories(colgm_mlir_core PUBLIC
    "${CMAKE_SOURCE_DIR}/src"
)

# platform-specific linkage
if(NOT WIN32)
    target_link_libraries(colgm_mlir_core PUBLIC pthread)
endif()

# ---- colgm mlir executable ----
add_executable(colgm-mlir src/main.cpp)

target_include_directories(colgm-mlir PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(colgm-mlir PRIVATE colgm_mlir_core)

# ---- type_storage stress test ----
add_executable(type_storage_test src/test/type_storage_test.cpp)
target_include_directories(type_storage_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(type_storage_test PRIVATE colgm_mlir_core)

if(ENABLE_MLIR)
    target_include_directories(colgm_mlir_core PUBLIC
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
    target_compile_definitions(colgm_mlir_core PUBLIC
        ${LLVM_DEFINITIONS}
    )
    target_compile_options(colgm_mlir_core PUBLIC
        -fno-rtti
    )
    target_link_libraries(colgm_mlir_core PUBLIC
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRDialect
        MLIRFuncDialect
        MLIRPass
    )

    # Standalone colgm-opt tool (like mlir-opt, but statically includes our dialect)
    add_executable(colgm-opt
        src/tools/colgm-opt.cpp
    )
    target_include_directories(colgm-opt PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
    target_compile_definitions(colgm-opt PRIVATE
        ${LLVM_DEFINITIONS}
    )
    target_compile_options(colgm-opt PRIVATE
        -fno-rtti
    )
    target_link_libraries(colgm-opt PRIVATE
        colgm_mlir_core
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRDialect
        MLIRFuncDialect
        MLIRTensorDialect
        MLIROptLib
    )
endif()

# ---- install ----
install(TARGETS colgm_mlir_core colgm-mlir
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)
