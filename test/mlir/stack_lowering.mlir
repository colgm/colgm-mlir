// Test lowering of colgm.stack to tensor ops

// Stack 1D tensors into 2D
func.func @test_stack_1d_to_2d(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<3x4xf32> {
  %0 = colgm.stack(%a, %b, %c) {target_shape = [3]} : tensor<4xf32> -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// Stack 2D tensors into 3D
func.func @test_stack_2d_to_3d(%a: tensor<2x4xi64>, %b: tensor<2x4xi64>) -> tensor<2x2x4xi64> {
  %0 = colgm.stack(%a, %b) {target_shape = [2]} : tensor<2x4xi64> -> tensor<2x2x4xi64>
  return %0 : tensor<2x2x4xi64>
}
