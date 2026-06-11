// verify error: batch dims mismatch
func.func @test_matmul_bad_batch(%arg0: tensor<2x3x4xf32>, %arg1: tensor<5x4x24xf32>) -> tensor<2x3x24xf32> {
  %0 = colgm.matmul %arg0, %arg1 : tensor<2x3x4xf32>, tensor<5x4x24xf32>
  return %0 : tensor<2x3x24xf32>
}
