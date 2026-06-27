// colgm dialect test: stack op — bad case (operand count mismatch)
// parse -> print -> verify should fail: 2 operands != 3 (product of target_shape)

func.func @bad_count(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<3x4xf32> {
  %0 = colgm.stack(%a, %b) {target_shape = [3]} : tensor<4xf32> -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
