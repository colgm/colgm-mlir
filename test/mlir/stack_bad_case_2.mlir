// colgm dialect test: stack op — bad case (missing target_shape)
// parse -> print -> verify should fail: no target_shape attribute

func.func @bad_missing_attr(%a: tensor<2x4xi64>) -> tensor<2x4xi64> {
  %0 = colgm.stack(%a) : tensor<2x4xi64> -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}
