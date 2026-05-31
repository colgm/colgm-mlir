// colgm dialect test: log op
// parse -> print -> verify should all pass

func.func @test_log(%arg: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %0 = colgm.log %arg : tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
