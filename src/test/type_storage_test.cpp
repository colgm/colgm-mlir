#include "sema/storage.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

namespace colgm_mlir {

#define TEST(name) \
    static bool test_##name(type_storage&); \
    static bool test_##name(type_storage& s)

#define RUN(name) \
    do { \
        fprintf(stderr, " - %-35s", #name); \
        bool ok = test_##name(s); \
        fprintf(stderr, " %s\n", ok ? "PASS" : "FAIL"); \
        if (!ok) all_pass = false; \
    } while (0)

TEST(uniquing_scalars) {
    // Each scalar getter must always return the exact same pointer.
    type a1 = s.get_i32_type();
    type a2 = s.get_i32_type();
    assert(a1 == a2);

    type b1 = s.get_i64_type();
    type b2 = s.get_i64_type();
    assert(b1 == b2);

    type c1 = s.get_f32_type();
    type c2 = s.get_f32_type();
    assert(c1 == c2);

    type d1 = s.get_f64_type();
    type d2 = s.get_f64_type();
    assert(d1 == d2);

    type e1 = s.get_bool_type();
    type e2 = s.get_bool_type();
    assert(e1 == e2);

    type f1 = s.get_void_type();
    type f2 = s.get_void_type();
    assert(f1 == f2);

    type g1 = s.get_unknown_type();
    type g2 = s.get_unknown_type();
    assert(g1 == g2);

    // Different scalars must NOT be equal.
    assert(a1 != b1);
    assert(a1 != c1);
    assert(a1 != e1);
    assert(f1 != e1);
    assert(g1 != f1);

    return true;
}

TEST(uniquing_functions) {
    // Same (args, ret) inserted twice → must be pointer-equal.
    auto i32 = s.get_i32_type();
    auto f64 = s.get_f64_type();

    auto fn1 = s.get_function_type({i32, f64}, i32);
    auto fn2 = s.get_function_type({i32, f64}, i32);
    assert(fn1 == fn2);

    // Different return type → not equal.
    auto fn3 = s.get_function_type({i32, f64}, f64);
    assert(fn1 != fn3);

    // Different arg order → not equal.
    auto fn4 = s.get_function_type({f64, i32}, i32);
    assert(fn1 != fn4);

    return true;
}

TEST(uniquing_tensors) {
    auto f32 = s.get_f32_type();
    std::vector<i64> shape1 = {2, 3};
    std::vector<i64> shape2 = {2, 3};

    auto t1 = s.get_tensor_type(f32, shape1);
    auto t2 = s.get_tensor_type(f32, shape2);
    assert(t1 == t2);

    // Different shape → not equal.
    auto t3 = s.get_tensor_type(f32, {3, 2});
    assert(t1 != t3);

    // Different element type, same shape → not equal.
    auto t4 = s.get_tensor_type(s.get_i64_type(), {2, 3});
    assert(t1 != t4);

    return true;
}

TEST(empty_args) {
    // fn() -> i32
    auto i32_ret = s.get_function_type({}, s.get_i32_type());
    assert(i32_ret.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION);
    assert(i32_ret == s.get_function_type({}, s.get_i32_type()));

    // fn() -> void
    auto void_ret = s.get_function_type({}, s.get_void_type());
    assert(void_ret.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION);
    assert(void_ret != i32_ret);

    return true;
}

TEST(many_args) {
    // 100 arguments, all i32.
    std::vector<type> args(100, s.get_i32_type());
    auto fn = s.get_function_type(args, s.get_void_type());
    assert(fn.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION);

    // Same 100 args again → uniqued.
    auto fn2 = s.get_function_type(args, s.get_void_type());
    assert(fn == fn2);

    // 99 args (one fewer) → different type.
    args.pop_back();
    auto fn3 = s.get_function_type(args, s.get_void_type());
    assert(fn != fn3);

    return true;
}

TEST(nested_function_types) {
    // fn(fn(i32)->f32, tensor<f32, 2, 3>) -> tensor<f32, 4>
    auto i32 = s.get_i32_type();
    auto f32 = s.get_f32_type();

    auto inner_fn = s.get_function_type({i32}, f32);
    assert(inner_fn.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION);

    auto tensor_2x3 = s.get_tensor_type(f32, {2, 3});
    auto tensor_4   = s.get_tensor_type(f32, {4});

    auto outer_fn = s.get_function_type({inner_fn, tensor_2x3}, tensor_4);
    assert(outer_fn.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION);

    // Same composite type again → uniqued.
    auto outer_fn2 = s.get_function_type({inner_fn, tensor_2x3}, tensor_4);
    assert(outer_fn == outer_fn2);

    return true;
}

TEST(tensor_extremes) {
    auto f32 = s.get_f32_type();

    // 0-d tensor: empty shape.
    auto t0 = s.get_tensor_type(f32, {});
    assert(t0.get_kind() == colgm_type_kind::TYPE_KIND_TENSOR);
    auto t0b = s.get_tensor_type(f32, {});
    assert(t0 == t0b);

    // 1-d tensor with large dimension.
    auto t1 = s.get_tensor_type(f32, {2147483647});
    assert(t1.get_kind() == colgm_type_kind::TYPE_KIND_TENSOR);
    assert(t1 == s.get_tensor_type(f32, {2147483647}));
    assert(t1 != t0);

    // High-rank tensor (6-d).
    auto t6 = s.get_tensor_type(f32, {1, 2, 3, 4, 5, 6});
    assert(t6.get_kind() == colgm_type_kind::TYPE_KIND_TENSOR);
    assert(t6 == s.get_tensor_type(f32, {1, 2, 3, 4, 5, 6}));

    return true;
}

TEST(void_return) {
    auto i32 = s.get_i32_type();
    auto f64 = s.get_f64_type();
    auto void_ty = s.get_void_type();

    auto fn = s.get_function_type({i32, f64}, void_ty);
    assert(fn.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION);
    assert(fn == s.get_function_type({i32, f64}, void_ty));

    // Different from non-void return.
    auto fn2 = s.get_function_type({i32, f64}, i32);
    assert(fn != fn2);

    return true;
}

TEST(hash_no_false_collision) {
    auto i32 = s.get_i32_type();
    auto f64 = s.get_f64_type();

    // fn(i32, f64) vs fn(f64, i32) — different arg order, must not collide.
    auto fn_a = s.get_function_type({i32, f64}, i32);
    auto fn_b = s.get_function_type({f64, i32}, i32);
    assert(fn_a != fn_b);

    // Same idea for tensors: tensor<i32, 2, 3> vs tensor<i32, 3, 2>.
    auto t_a = s.get_tensor_type(i32, {2, 3});
    auto t_b = s.get_tensor_type(i32, {3, 2});
    assert(t_a != t_b);

    return true;
}

TEST(volume) {
    // Insert 10000 unique function types.
    type scalars[6] = {
        s.get_i32_type(),
        s.get_i64_type(),
        s.get_f32_type(),
        s.get_f64_type(),
        s.get_bool_type(),
        s.get_void_type()
    };

    for (int i = 0; i < 50000; i++) {
        int n = i;
        std::vector<type> args;
        args.reserve(12);
        for (int j = 0; j < 12; j++) {
            args.push_back(scalars[n % 6]);
            n /= 6;
        }
        auto fn = s.get_function_type(args, scalars[i % 6]);
        assert(fn.get_kind() == colgm_type_kind::TYPE_KIND_FUNCTION);
    }

    // Insert 10000 unique tensor types.
    // Rank-1 tensors with varying single dimension: tensor<f32, [i]>.
    auto f32 = s.get_f32_type();
    for (int i = 0; i < 50000; i++) {
        auto t = s.get_tensor_type(f32, {static_cast<i64>(i)});
        assert(t.get_kind() == colgm_type_kind::TYPE_KIND_TENSOR);
    }

    return true;
}

TEST(destructor) {
    // Create types in a nested scope, ensure destructor handles cleanup.
    {
        type_storage local;
        auto i32 = local.get_i32_type();
        auto f32 = local.get_f32_type();

        // Insert a mix of function and tensor types.
        for (int i = 0; i < 100; i++) {
            local.get_function_type({i32, f32}, i32);
            local.get_tensor_type(f32, {static_cast<i64>(i)});
        }
        // local goes out of scope here — run under valgrind/ASAN to
        // verify no leaks or double-frees.
    }
    return true;
}

}  // namespace colgm_mlir

int main() {
    using namespace colgm_mlir;

    fprintf(stderr, "=== type_storage stress test ===\n\n");

    type_storage s;
    bool all_pass = true;

    RUN(uniquing_scalars);
    RUN(uniquing_functions);
    RUN(uniquing_tensors);
    RUN(empty_args);
    RUN(many_args);
    RUN(nested_function_types);
    RUN(tensor_extremes);
    RUN(void_return);
    RUN(hash_no_false_collision);
    RUN(volume);
    RUN(destructor);

    fprintf(stderr, "\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
