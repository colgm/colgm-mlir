// colgm dialect test: cast op — bad case (scalar ↔ tensor)
// parse -> print -> verify should fail: scalar and tensor cannot cross-cast

func.func @bad_scalar_tensor(%arg: i64) -> tensor<f32> {
  %0 = colgm.cast %arg : i64 -> tensor<f32>
  return %0 : tensor<f32>
}
