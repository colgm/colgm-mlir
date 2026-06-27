// colgm dialect test: stack op — bad case (shape mismatch)
// parse -> print -> verify should fail: result shape != target_shape + operand_shape

func.func @bad_shape(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<2x2xf32> {
  %0 = colgm.stack(%a, %b) {target_shape = [2]} : tensor<4xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
