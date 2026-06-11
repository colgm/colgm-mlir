// colgm dialect test: batch matmul (3D)
// parse -> print -> verify should all pass
func.func @test_batch_matmul(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x24xf32>) -> tensor<2x3x24xf32> {
  %0 = colgm.matmul %arg0, %arg1 : tensor<2x3x4xf32>, tensor<2x4x24xf32>
  return %0 : tensor<2x3x24xf32>
}
