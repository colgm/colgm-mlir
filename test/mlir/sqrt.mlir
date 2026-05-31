// colgm dialect test: sqrt op
// parse -> print -> verify should all pass

func.func @test_sqrt(%arg: tensor<3xf32>) -> tensor<3xf32> {
  %0 = colgm.sqrt %arg : tensor<3xf32>
  return %0 : tensor<3xf32>
}
