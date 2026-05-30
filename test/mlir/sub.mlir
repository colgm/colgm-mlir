// colgm dialect test: sub op
// parse -> print -> verify should all pass

// scalar add (rank-0 tensor)
func.func @sub_scalar(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = colgm.sub %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// vector add (rank-1 tensor)
func.func @sub_vector(%arg0: tensor<5xi32>, %arg1: tensor<5xi32>) -> tensor<5xi32> {
  %0 = colgm.sub %arg0, %arg1 : tensor<5xi32>
  return %0 : tensor<5xi32>
}

// matrix add (rank-2 tensor)
func.func @sub_matrix(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = colgm.sub %arg0, %arg1 : tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}
