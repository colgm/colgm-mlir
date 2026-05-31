// colgm dialect test: neg op
// parse -> print -> verify should all pass

func.func @test_neg(%arg: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = colgm.neg %arg : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
