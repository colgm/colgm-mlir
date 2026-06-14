// colgm dialect test: elements op — bad case (shape mismatch)
// parse -> print -> verify should fail: target_shape != result shape

func.func @bad_shape(%a: f32, %b: f32, %c: f32, %d: f32) -> tensor<2x2xf32> {
  %0 = colgm.elements(%a, %b, %c, %d) {target_shape = [4]} : f32 -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
