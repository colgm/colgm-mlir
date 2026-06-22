// verify error: then/else yield different types
func.func @bad_yield_type_mismatch(%cond: i1, %a: tensor<f32>, %b: tensor<i64>) -> tensor<f32> {
  %0 = colgm.if %cond -> tensor<f32> {
    colgm.yield %a : tensor<f32>
  } else {
    colgm.yield %b : tensor<i64>
  }
  func.return %0 : tensor<f32>
}
