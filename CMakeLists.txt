cmake_minimum_required(VERSION 3.20)
project(colgm-mlir
    VERSION 0.4.1
    DESCRIPTION "Experimental MLIR-based compiler"
    LANGUAGES CXX
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
        include_directories(${LLVM_INCLUDE_DIRS})
        include_directories(${MLIR_INCLUDE_DIRS})
        add_definitions(${LLVM_DEFINITIONS})
        message(STATUS "MLIR support enabled (LLVM ${LLVM_PACKAGE_VERSION})")
    else()
        message(FATAL_ERROR "ENABLE_MLIR=ON but LLVM_DIR or MLIR_DIR not found. "
                            "Run scripts/download_llvm.py first.")
    endif()
endif()

# ---- source files ----
set(COLOGM_SOURCES
    src/ast/ast.cpp
    src/main.cpp
    src/misc.cpp
    src/lexer.cpp
    src/report.cpp
)

# ---- colgm mlir core library ----
add_library(colgm_mlir_core STATIC
    src/ast/ast.cpp
    src/misc.cpp
    src/lexer.cpp
    src/report.cpp
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

if(ENABLE_MLIR)
    target_link_libraries(colgm_mlir_core PRIVATE
        ${LLVM_LIBRARIES}
        ${MLIR_LIBRARIES}
    )
    target_link_libraries(colgm-mlir PRIVATE
        ${LLVM_LIBRARIES}
        ${MLIR_LIBRARIES}
    )
endif()

# ---- install ----
install(TARGETS colgm_mlir_core colgm-mlir
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)

# ---- version defines ----
target_compile_definitions(colgm_mlir_core PRIVATE
    __colgm_ver__="${PROJECT_VERSION}"
)

target_compile_definitions(colgm-mlir PRIVATE
    __colgm_ver__="${PROJECT_VERSION}"
)
