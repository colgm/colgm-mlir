// verify error: tensor rank must be equal
func.func @test_matmul_bad_rank2(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4x24xf32>) -> tensor<2x24xf32> {
  %0 = colgm.matmul %arg0, %arg1 : tensor<2x4xf32>, tensor<3x4x24xf32>
  return %0 : tensor<2x24xf32>
}
