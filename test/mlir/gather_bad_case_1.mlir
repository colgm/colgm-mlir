// colgm dialect test: gather op — bad case (axis out of range)

func.func @bad_gather_axis(%params: tensor<4x3xf64>, %indices: tensor<2xi64>) -> tensor<2x3xf64> {
  %0 = colgm.gather %params, %indices {axis = 3} : tensor<4x3xf64>, tensor<2xi64> -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}
