// colgm dialect test: gather op — bad case (missing axis)

func.func @bad_gather(%params: tensor<4x3xf64>, %indices: tensor<2xi64>) -> tensor<2x3xf64> {
  %0 = colgm.gather %params, %indices : tensor<4x3xf64>, tensor<2xi64> -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
