// colgm dialect test: elements op
// parse -> print -> verify should all pass

func.func @test_1d(%a: f32, %b: f32, %c: f32) -> tensor<3xf32> {
  %0 = colgm.elements(%a, %b, %c) {target_shape = [3]} : f32 -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

func.func @test_2d(%a: f64, %b: f64, %c: f64, %d: f64) -> tensor<2x2xf64> {
  %0 = colgm.elements(%a, %b, %c, %d) {target_shape = [2, 2]} : f64 -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
