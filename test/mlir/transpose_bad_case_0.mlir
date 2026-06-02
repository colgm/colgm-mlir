// verify error: element type mismatch
func.func @test_transpose_bad(%arg: tensor<2x3xf32>) -> tensor<3x2xi32> {
  %0 = colgm.transpose %arg {permutation = [1, 0]} : tensor<2x3xf32> -> tensor<3x2xi32>
  return %0 : tensor<3x2xi32>
}
