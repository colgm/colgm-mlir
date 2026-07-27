// colgm dialect test: gather op (embedding lookup)
// parse -> print -> verify should all pass

func.func @test_gather(%params: tensor<4x3xf64>, %indices: tensor<2xi64>) -> tensor<2x3xf64> {
  %0 = colgm.gather %params, %indices {axis = 0} : tensor<4x3xf64>, tensor<2xi64> -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
