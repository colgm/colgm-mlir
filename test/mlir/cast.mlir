// colgm dialect test: cast op
// parse -> print -> verify should all pass

func.func @scalar_int_index(%arg: i64) -> index {
  %0 = colgm.cast %arg : i64 -> index
  return %0 : index
}

func.func @scalar_float_width(%arg: f64) -> f32 {
  %0 = colgm.cast %arg : f64 -> f32
  return %0 : f32
}

func.func @tensor_element_cast(%arg: tensor<2x3xf64>) -> tensor<2x3xf32> {
  %0 = colgm.cast %arg : tensor<2x3xf64> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
