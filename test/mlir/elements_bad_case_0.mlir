// colgm dialect test: elements op — bad case (type mismatch)
// parse -> print -> verify should fail: operands have different types

func.func @bad_mixed_types(%a: tensor<f32>, %b: tensor<f64>) -> tensor<2xf32> {
  %0 = colgm.elements(%a, %b) {target_shape = [2]} : tensor<f32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
