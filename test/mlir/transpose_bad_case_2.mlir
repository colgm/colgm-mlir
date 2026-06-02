// verify error: permutation size != input rank
func.func @test_transpose_bad(%arg: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = colgm.transpose %arg {permutation = [0, 1]} : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}
