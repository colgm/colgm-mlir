// verify error: missing permutation attribute
func.func @test_transpose_bad(%arg: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = colgm.transpose %arg : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
