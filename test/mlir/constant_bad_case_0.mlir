// verify error: dense type mismatch (2 elements vs 3)
func.func @bad_dense_shape() -> tensor<3xi64> {
  %0 = colgm.constant dense<[1, 2]> : tensor<3xi64>
  func.return %0 : tensor<3xi64>
}
