// verify error
func.func @test_reshape_bad(%arg: tensor<2x3xf32>) -> tensor<2x2xf32> {
  %0 = colgm.reshape %arg {target_shape = [2, 2]} : tensor<2x3xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
