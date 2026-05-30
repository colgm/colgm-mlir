// colgm dialect test: mul op
// parse -> print -> verify should all pass

// scalar mul (rank-0 tensor)
func.func @mul_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = colgm.mul %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// vector mul (rank-1 tensor)
func.func @mul_vector(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = colgm.mul %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// matrix mul (rank-2 tensor)
func.func @mul_matrix(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = colgm.mul %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
