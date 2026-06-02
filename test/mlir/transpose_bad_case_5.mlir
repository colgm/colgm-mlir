// verify error: output shape mismatch
func.func @test_transpose_bad(%arg: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = colgm.transpose %arg {permutation = [1, 0]} : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
