// colgm dialect test: abs op
// parse -> print -> verify should all pass

func.func @test_abs(%arg: tensor<4xf64>) -> tensor<4xf64> {
  %0 = colgm.abs %arg : tensor<4xf64>
  return %0 : tensor<4xf64>
}
