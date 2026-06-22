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

// rank-0 tensor to scalar
func.func @rank0_tensor_to_scalar(%arg: tensor<i1>) -> i1 {
  %0 = colgm.cast %arg : tensor<i1> -> i1
  return %0 : i1
}

// scalar to rank-0 tensor
func.func @scalar_to_rank0_tensor(%arg: i1) -> tensor<i1> {
  %0 = colgm.cast %arg : i1 -> tensor<i1>
  return %0 : tensor<i1>
}

// rank-0 tensor to scalar with int->index cast
func.func @rank0_tensor_int_to_index(%arg: tensor<i64>) -> index {
  %0 = colgm.cast %arg : tensor<i64> -> index
  return %0 : index
}

// scalar index to rank-0 tensor int
func.func @index_to_rank0_tensor_int(%arg: index) -> tensor<i64> {
  %0 = colgm.cast %arg : index -> tensor<i64>
  return %0 : tensor<i64>
}
