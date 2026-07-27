// colgm dialect test: reduce_max op
// parse -> print -> verify should all pass

func.func @test_reduce_max(%arg: tensor<3x4xf64>) -> tensor<4xf64> {
  %0 = colgm.reduce_max %arg {axes = [0]} : tensor<3x4xf64> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}
