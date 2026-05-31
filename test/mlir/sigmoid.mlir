// colgm dialect test: sigmoid op
// parse -> print -> verify should all pass

func.func @test_sigmoid(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = colgm.sigmoid %arg : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
