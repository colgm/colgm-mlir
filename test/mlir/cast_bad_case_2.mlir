// colgm dialect test: cast op — bad case (scalar → rank-0 tensor, incompatible element type)
// parse -> print -> verify should fail: i64 cannot cast to f32 element

func.func @bad_scalar_tensor(%arg: i64) -> tensor<f32> {
  %0 = colgm.cast %arg : i64 -> tensor<f32>
  return %0 : tensor<f32>
}
