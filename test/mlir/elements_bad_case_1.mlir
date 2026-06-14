// colgm dialect test: elements op — bad case (element count mismatch)
// parse -> print -> verify should fail: 2 operands != 3 elements

func.func @bad_count(%a: f32, %b: f32) -> tensor<3xf32> {
  %0 = colgm.elements(%a, %b) {target_shape = [3]} : f32 -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
