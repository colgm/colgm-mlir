// colgm dialect test: exp op
// parse -> print -> verify should all pass

func.func @test_exp(%arg: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = colgm.exp %arg : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
