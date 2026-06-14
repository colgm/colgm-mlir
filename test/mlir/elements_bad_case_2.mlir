// colgm dialect test: elements op — bad case (missing target_shape)
// parse -> print -> verify should fail: no target_shape attribute

func.func @bad_missing_attr(%a: f32) -> tensor<f32> {
  %0 = colgm.elements(%a) : f32 -> tensor<f32>
  return %0 : tensor<f32>
}
