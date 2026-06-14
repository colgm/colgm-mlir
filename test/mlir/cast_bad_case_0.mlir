// colgm dialect test: cast op — bad case (cross-kind)
// parse -> print -> verify should fail: int -> float not allowed

func.func @bad_int_to_float(%arg: i64) -> f32 {
  %0 = colgm.cast %arg : i64 -> f32
  return %0 : f32
}
