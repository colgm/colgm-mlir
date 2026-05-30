// colgm dialect test: div op
// parse -> print -> verify should all pass

// scalar div (rank-0 tensor)
func.func @div_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = colgm.div %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// vector div (rank-1 tensor)
func.func @div_vector(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = colgm.div %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// matrix div (rank-2 tensor)
func.func @div_matrix(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = colgm.div %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
