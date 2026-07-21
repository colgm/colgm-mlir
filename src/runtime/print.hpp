#pragma once

#include <cstdint>

extern "C" {
void __colgm_print_f64(double v);
void __colgm_print_f32(float v);
void __colgm_print_i64(int64_t v);
void __colgm_print_i32(int32_t v);
void __colgm_print_i1(bool v);
void __colgm_print_open_bracket(void);
void __colgm_print_close_bracket(void);
void __colgm_print_comma(void);
void __colgm_print_newline(void);
}