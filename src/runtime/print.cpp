#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

extern "C" {
void __colgm_print_f64(double v) { printf("%g", v); }

void __colgm_print_f32(float v) { printf("%g", (double)v); }

void __colgm_print_i64(int64_t v) { printf("%ld", v); }

void __colgm_print_i32(int32_t v) { printf("%d", v); }

void __colgm_print_i1(bool v) { printf(v ? "true" : "false"); }

void __colgm_print_open_bracket(void) { printf("["); }

void __colgm_print_close_bracket(void) { printf("]"); }

void __colgm_print_comma(void) { printf(", "); }

void __colgm_print_newline(void) { printf("\n"); }
}
