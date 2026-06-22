// verify error: then/else yield different number of values
func.func @bad_yield_count_mismatch(%cond: i1, %a: tensor<f32>) -> tensor<f32> {
  %0 = colgm.if %cond -> tensor<f32> {
    colgm.yield %a : tensor<f32>
  } else {
    colgm.yield
  }
  func.return %0 : tensor<f32>
}
