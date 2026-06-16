// colgm dialect test: cmp_eq — bad case (result shape mismatch)
// parse -> print -> verify should fail

func.func @bad_result_shape(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<4x5xi1> {
  %0 = colgm.cmp_eq %a, %b : tensor<2x3xf32> -> tensor<4x5xi1>
  func.return %0 : tensor<4x5xi1>
}
