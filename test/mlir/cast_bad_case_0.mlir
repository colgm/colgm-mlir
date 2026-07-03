// colgm dialect test: cast op — bad case (tensor shape mismatch)
// parse -> print -> verify should fail: shapes differ

func.func @bad_shape(%arg: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = colgm.cast %arg : tensor<2x3xf32> -> tensor<6xf32>
  return %0 : tensor<6xf32>
}
