// colgm dialect test: tanh op
// parse -> print -> verify should all pass

func.func @test_tanh(%arg: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %0 = colgm.tanh %arg : tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
