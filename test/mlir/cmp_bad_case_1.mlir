// colgm dialect test: cmp_eq — bad case (result element type not i1)
// parse -> print -> verify should fail

func.func @bad_result_type(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
  %0 = colgm.cmp_eq %a, %b : tensor<f32> -> tensor<f32>
  func.return %0 : tensor<f32>
}
