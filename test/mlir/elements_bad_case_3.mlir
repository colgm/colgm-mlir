// colgm dialect test: elements op — bad case (shape mismatch)
// parse -> print -> verify should fail: target_shape != result shape

func.func @bad_shape(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>, %d: tensor<f32>) -> tensor<2x2xf32> {
  %0 = colgm.elements(%a, %b, %c, %d) {target_shape = [4]} : tensor<f32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
