// colgm dialect test: gelu op
// parse -> print -> verify should all pass

func.func @test_gelu(%arg: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %0 = colgm.gelu %arg : tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
