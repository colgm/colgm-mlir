// verify error: tensor rank must be >= 2
func.func @test_matmul_bad_rank(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4x4xf32> {
  %0 = colgm.matmul %arg0, %arg1 : tensor<4xf32>, tensor<4xf32>
  return %0 : tensor<4x4xf32>
}
