# Syntax Design

## Overview

A static-shape tensor computation DSL with C/Rust-style infix syntax,
Go-style automatic semicolon insertion, targeting MLIR as the compiler backend.

---

## 1. Data Types

### Scalar Types

| Keyword | Meaning |
|---------|---------|
| `f32`   | 32-bit floating point |
| `f64`   | 64-bit floating point |
| `i32`   | 32-bit signed integer |
| `i64`   | 64-bit signed integer |
| `bool`  | boolean (`true` / `false`) |

### Tensor Types

A tensor type is written as `dtype[dim0, dim1, ...]`. All dimensions are
compile-time integer constants (static shapes only).

```rs
f32[2, 3]       // 2x3 matrix of f32
i64[4]           // 4-element vector of i64
f64[2, 2, 2]    // 2x2x2 3-D tensor of f64
```

A scalar is just a zero-dimensional tensor: `f32[]` or simply `f32`.

---

## 2. Comments

```rs
// single-line comment

/*
   multi-line
   comment
*/
```

---

## 3. Literals

### Scalar Literals

```rs
42               // i64 (default integer)
3.14             // f64 (default float)
true             // bool
false            // bool
```

Integer literals support hex and octal prefixes:

```rs
0x7fffffff       // hex → i64
0o777            // octal → i64
```

### Tensor Literals

Nested-bracket syntax. The shape and element type are inferred:

```rs
[[1.0, 2.0], [3.0, 4.0]]       // f64[2, 2]
[[1, 2, 3], [4, 5, 6]]          // i64[2, 3]
[1.0, 2.0, 3.0]                 // f64[3]
```

Type can be made explicit via a variable annotation:

```rs
var a: f32[2, 2] = [[1.0, 2.0], [3.0, 4.0]]
```

---

## 4. Variables

Declared with `var`. Type annotation is optional when the initializer
can infer the type:

```rs
var x = [[1.0, 2.0], [3.0, 4.0]]       // type inferred: f64[2, 2]
var y: f32[2, 2] = [[1.0, 2.0], [3.0, 4.0]]
var z = a + b                            // type inferred from a, b
```

Assignment to an existing variable (no `var`):

```rs
x = c * 2.0
```

---

## 5. Operators (arithmetic)

Arithmetic uses infix operators. Semantics are always element-wise.

| Operator | Meaning |
|----------|---------|
| `+`      | element-wise add |
| `-`      | element-wise subtract (binary), negate (unary) |
| `*`      | element-wise multiply |
| `/`      | element-wise divide |
| `==`     | element-wise equal |
| `!=`     | element-wise not equal |
| `<`      | element-wise less than |
| `>`      | element-wise greater than |
| `<=`     | element-wise less or equal |
| `>=`     | element-wise greater or equal |

All operands must have the same shape (no broadcasting in the initial version).

### Precedence (lowest → highest)

| Level | Operators |
|-------|-----------|
| 1     | `==` `!=` `<` `>` `<=` `>=` |
| 2     | `+` `-` |
| 3     | `*` `/` |
| 4     | unary `-` |

Parentheses `(...)` override precedence as usual.

---

## 6. Function Calls (all other ops)

Everything beyond basic arithmetic is a function-style call with optional
named parameters:

```rs
relu(x)                   // activation: element-wise max(0, x)
sigmoid(x)                // activation: 1 / (1 + exp(-x))
tanh(x)                   // activation: tanh
exp(x)                    // element-wise e^x
log(x)                    // element-wise ln(x)
sqrt(x)                   // element-wise sqrt
abs(x)                    // element-wise |x|

matmul(a, b)              // matrix multiplication

reshape(a, [2, 4])        // reshape to new dimensions
transpose(a, [1, 0])      // permute axes
sum(a, 0)                 // reduce along axis
mean(a, 1)                // reduce along axis
```

The function-call syntax `name(arg, param=val, ...)` is used for:
- All non-arithmetic tensor operations
- User-defined functions

---

## 7. Functions

Defined with the `func` keyword. Parameters and return type use tensor types:

```rs
func add_tensors(x: f32[2, 2], y: f32[2, 2]) -> f32[2, 2] {
    return x + y
}
```

A function with no return value omits the `->` clause:

```rs
func print_debug(x: f32[2, 2]) {
    // ...
}
```

The entry point is a function named `main`:

```rs
func main() {
    var a = [[1.0, 2.0], [3.0, 4.0]]
    var b = relu(a)
}
```

---

## 8. Control Flow

### if / else

```rs
if x == y {
    // ...
} else {
    // ...
}
```

No `elsif` keyword. Nested `if` / `else` only:

```rs
if a {
    // ...
} else if b {
    // ...
} else {
    // ...
}
```

### for (range-based)

```rs
for i in 0..10 {
    // i is i64, values 0 through 9
}

for i in 0..N {
    // N can be a variable
}
```

### return

```rs
return expr
return                       // for void functions
```

---

## 9. Automatic Semicolon Insertion (Go-style)

Statements are separated by newlines, not `;` tokens. The lexer
automatically inserts a virtual semicolon at a line break when the
line ends with a token that can terminate a statement:

**Semicolon is inserted** after a line ending with:
- An identifier
- A literal (`42`, `3.14`, `true`, `false`)
- A closing bracket `]`, `)`, `}`
- The keyword `return` (if followed by a line break with no expression — handled in the parser)

**Semicolon is NOT inserted** (line continues) when the line ends with:
- An operator (`+`, `-`, `*`, `/`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `=`)
- An opening bracket `[`, `(`, `{`
- A comma `,`
- A dot `.`

This means multi-line expressions and calls work naturally:

```rs
var a = x +
        y +
        z                   // one statement: x + y + z

var b = relu(               // one statement: relu(x)
    x
)

var c = [[1, 2],
         [3, 4]]            // one statement: 2x2 tensor

var d = reshape(a, [2, 4])  // one statement: reshape call
```

---

## 10. Complete Example

```rs
func relu(x: f32[2, 2]) -> f32[2, 2] {
    return max(x, 0.0)
}

func main() {
    var a = [[1.0, -2.0], [-3.0, 4.0]]
    var b = relu(a)
    var c = b + [[1.0, 0.0], [0.0, 1.0]]
    var d = reshape(c, [4])

    for i in 0..4 {
        var val = exp(d[i])
    }
}
```

---

## 11. What Is NOT in Scope (for now)

- **No structs** — functions are the only top-level organization unit
- **No modules / imports** — all code in a single source file
- **No generics** — no type parameters on functions
- **No dynamic shapes** — every `dtype[dim, ...]` has literal-integer dimensions
- **No broadcasting** — element-wise ops require identical shapes
- **No closures / lambdas** — only top-level named functions
- **No enums / tagged unions** — just tensors and scalars
- **No pointers / references** — value semantics only
- **No `defer`** — plain return-based control flow
