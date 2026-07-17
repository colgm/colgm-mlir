// Test lowering of colgm.elements to tensor ops

// 1D case: no reshape needed
func.func @test_elements_1d(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>) -> tensor<3xf32> {
  %0 = colgm.elements(%a, %b, %c) {target_shape = [3]} : tensor<f32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// 2D case: requires tensor.reshape
func.func @test_elements_2d(%a: tensor<f64>, %b: tensor<f64>, %c: tensor<f64>, %d: tensor<f64>) -> tensor<2x2xf64> {
  %0 = colgm.elements(%a, %b, %c, %d) {target_shape = [2, 2]} : tensor<f64> -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
