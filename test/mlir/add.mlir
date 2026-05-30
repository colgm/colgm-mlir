// colgm dialect test: add op
// parse -> print -> verify should all pass

// scalar add (rank-0 tensor)
func.func @add_scalar(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
  %0 = colgm.add %a, %b : tensor<f32>
  func.return %0 : tensor<f32>
}

// tensor add (rank-2)
func.func @add_tensor(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = colgm.add %x, %y : tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
