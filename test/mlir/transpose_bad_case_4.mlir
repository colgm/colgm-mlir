// verify error: duplicate axis
func.func @test_transpose_bad(%arg: tensor<2x3x4xf32>) -> tensor<2x4x3xf32> {
  %0 = colgm.transpose %arg {permutation = [0, 1, 0]} : tensor<2x3x4xf32> -> tensor<2x4x3xf32>
  return %0 : tensor<2x4x3xf32>
}
