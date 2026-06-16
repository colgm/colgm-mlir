// colgm dialect test: cmp_eq — bad case (mismatched operand types)
// parse -> print -> verify should fail

func.func @bad_mismatched_types(%a: tensor<f32>, %b: tensor<i32>) -> tensor<i1> {
  %0 = colgm.cmp_eq %a, %b : tensor<f32> -> tensor<i1>
  func.return %0 : tensor<i1>
}
