// verify error
func.func @test_reshape_bad(%arg: tensor<2x3xf32>) -> tensor<3x2xi32> {
  %0 = colgm.reshape %arg {target_shape = [3, 2]} : tensor<2x3xf32> -> tensor<3x2xi32>
  return %0 : tensor<3x2xi32>
}
