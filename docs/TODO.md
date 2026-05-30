# TODO — colgm-mlir

## Design Decisions (settled)

- **Static shapes only** — all tensor dimensions are compile-time constants. MLIR types like `tensor<3x256x256xf32>`. No `?` / dynamic dims initially.
- **Graph-level auto-optimization** — users describe the computation graph; MLIR built-in passes (Linalg, Affine, SCF) handle tiling, fusion, and vectorization. No manual compute/schedule separation.
- **C/Rust-style infix syntax** — similar to the original Colgm, but computation statements redesigned for tensor DSL.
- **Lexer borrowed from Colgm bootstrap (C++)** — token kinds trimmed and keywords adjusted for the tensor domain.
- **Target backend** — MLIR → LLVM IR (via the existing `-convert-scf-to-cf`, `-convert-to-llvm` pipeline).

---

## Phase 0 — Project Skeleton

- [x] Set up CMake build (MLIR dependency, LLVM find_package)
- [x] Bring up minimal `main.cpp` that links against MLIR and prints the MLIR version
- [x] Choose the LLVM/MLIR version to track (llvm-project revision)

---

## Phase 1 — Lexer

- [x] Port Colgm bootstrap lexer (`lexer.cpp`, `lexer.h`) into the project
- [x] Slash the token enum down to what this DSL needs
  - Keywords: `func`, `var`, `return`, `if`, `else`, `for`, `struct`, `pub`, `extern`, `true`, `false`
  - Domain keywords: `tensor`, `f32`, `f64`, `i32`, `i64` (dtype keywords)
  - Literals: integer, float, string, char
  - Operators: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `=`, `->`, `(`, `)`, `{`, `}`, `[`, `]`, `,`, `;`, `.`, `:`
- [ ] Write smoke tests for the lexer (tokenize a few example `.toy` / `.dsl` files)

---

## Phase 2 — AST

- [x] Define the AST node hierarchy (similar to Colgm's `ast_kind` discriminated-union pattern, or a class hierarchy)
  - **Declarations**: `func_decl`, `struct_decl`
  - **Statements**: `var_decl`, `assign`, `if_stmt`, `for_stmt`, `return_stmt`, `expr_stmt`, `block`
  - **Expressions**: `binary`, `unary`, `call`, `tensor_literal`, `identifier`, `number_literal`, `float_literal`, `index_access`
- [x] Each AST node carries a `span` for error reporting
- [ ] AST dumper / pretty-printer for debugging

---

## Phase 3 — Parser

- [ ] Recursive-descent parser (hand-written, following Colgm's parser / MLIR Toy parser pattern)
- [ ] Parse function declarations: `func name(params...) -> ret_type { body }`
- [ ] Parse variable declarations: `var name = expr;` or `var name: type = expr;`
- [ ] Parse tensor type annotations: `f32[2, 3]`, `i32[4]`, `f64[2, 2, 2]`
- [ ] Parse expressions with precedence climbing (Pratt parser or layered recursive descent)
- [ ] Parse tensor literals: `[[1.0, 2.0], [3.0, 4.0]]`
- [ ] Parse control flow: `if`, `for`
- [ ] Error recovery strategy (at minimum: report the first error and exit, like Colgm bootstrap)

---

## Phase 4 — MLIR Dialect Definition

- [ ] Write TableGen files for the `tensor_ops` (or `toy2`) dialect
  - **ops**: `AddOp`, `SubOp`, `MulOp`, `DivOp`, `NegOp`, `ReluOp`, `ReshapeOp`
  - Each op has a custom assembly format and a verifier
- [ ] Define the `ToyTensorType` wrapper (or reuse MLIR's built-in `TensorType`)
- [ ] Write the `func` operation for the dialect (or reuse the built-in `func.func`)
- [ ] Register the dialect with MLIR

---

## Phase 5 — AST → MLIR Lowering

- [ ] Write `MLIRGen` visitor class (analogous to Toy tutorial's `MLIRGen`)
- [ ] Expression lowering: each AST expression node → MLIR `Value` (via op builder)
- [ ] Statement lowering: var declarations → `tensor_ops.constant` + SSA values
- [ ] Control flow lowering: `if` / `for` → `scf.if` / `scf.for`
- [ ] Function lowering: `func_decl` → `func.func` with body region
- [ ] Shape propagation: infer result tensor shapes from input shapes

---

## Phase 6 — MLIR Passes & Optimization

- [ ] Shape inference pass (if not fully resolved during lowering)
- [ ] Canonicalization patterns for each op (e.g., `AddOp(x, 0) → x`, constant folding)
- [ ] Inline pass (inline small functions)
- [ ] Run the standard MLIR lowering pipeline:
  - `-canonicalize`, `-cse`
  - `-convert-tensor-to-linalg` (or custom lowering)
  - `-convert-linalg-to-affine-loops`, `-lower-affine`
  - `-convert-scf-to-cf`, `-convert-to-llvm`

---

## Phase 7 — Code Generation (via MLIR → LLVM)

- [ ] Wire the `mlir-translate` / JIT pipeline
- [ ] Emit a `.ll` file or JIT-compile and run
- [ ] Provide a CLI: `colgm-mlir <file.dsl> [-o output.ll] [--run] [--dump-mlir]`

---

## Phase 8 — Operators: Batch 1 (element-wise)

- [ ] `add` — element-wise tensor addition (broadcasting not required yet)
- [ ] `sub` — element-wise tensor subtraction
- [ ] `mul` — element-wise tensor multiplication
- [ ] `div` — element-wise tensor division
- [ ] `neg` — element-wise negation
- [ ] `relu` — `max(0, x)` element-wise
- [ ] `sigmoid` — `1 / (1 + exp(-x))` element-wise
- [ ] `tanh` — element-wise hyperbolic tangent
- [ ] `exp` — element-wise exponential
- [ ] `log` — element-wise natural logarithm
- [ ] `sqrt` — element-wise square root
- [ ] `abs` — element-wise absolute value
- [ ] `reshape` — reshape tensor to new dimensions (total elements must match)
- [ ] `transpose` — permute tensor axes

---

## Phase 9 — Operators: Batch 2 (reductions + matmul)

- [ ] `sum` — reduce along specified axis (or all axes)
- [ ] `mean` — reduce along specified axis
- [ ] `max` — reduce along specified axis
- [ ] `min` — reduce along specified axis
- [ ] `matmul` — matrix multiplication (2-D only initially)
- [ ] `broadcast` — broadcast a tensor to a larger shape

---

## Phase 10 — Operators: Batch 3 (convolution & beyond)

- [ ] `conv2d` — 2-D convolution
- [ ] `max_pool2d` / `avg_pool2d` — 2-D pooling
- [ ] `gemm` — general matrix multiply (with optional transpose flags)

---

## Phase 11 — Polish & Ecosystem

- [ ] Better error messages (point to source location, show context)
- [ ] `std` library written in the DSL itself (math functions, initializers: `zeros`, `ones`, `rand`)
- [ ] Test suite: run each test case through the full pipeline and compare output values
- [ ] CI (GitHub Actions) with a pinned LLVM build
- [ ] Editor syntax highlighting (VSCode extension, similar to colgm-extension)

---

## Open Design Questions (to resolve as we go)

- Exact tensor literal syntax: `[[1,2],[3,4]]` vs `tensor([[1,2],[3,4]])` vs `f32[2,2]{1,2,3,4}`?
- Function-call style for ops vs operators: `a + b` vs `add(a, b)` vs both?
- Should `reshape` / `transpose` expose axis arguments inline or via a separate `@config` / annotation syntax?
- Does `for` loop stay general-purpose, or should it be specialized for tensor iteration (like `for i in 0..N`)?
