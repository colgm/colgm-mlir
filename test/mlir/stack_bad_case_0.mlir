// colgm dialect test: stack op — bad case (type mismatch)
// parse -> print -> verify should fail: operands have different types

func.func @bad_mixed_types(%a: tensor<2x4xi64>, %b: tensor<3x4xi64>) -> tensor<2x2x4xi64> {
  %0 = colgm.stack(%a, %b) {target_shape = [2]} : tensor<2x4xi64> -> tensor<2x2x4xi64>
  return %0 : tensor<2x2x4xi64>
}
