// colgm dialect test: constant op — positive cases

// scalar int
func.func @const_int() -> i64 {
  %0 = colgm.constant 42 : i64
  func.return %0 : i64
}

// scalar float
func.func @const_float() -> f32 {
  %0 = colgm.constant 3.14 : f32
  func.return %0 : f32
}

// scalar bool true
func.func @const_bool_true() -> i1 {
  %0 = colgm.constant true
  func.return %0 : i1
}

// scalar bool false
func.func @const_bool_false() -> i1 {
  %0 = colgm.constant false
  func.return %0 : i1
}

// dense 1D
func.func @const_dense_1d() -> tensor<3xi64> {
  %0 = colgm.constant dense<[1, 2, 3]> : tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// dense 2D
func.func @const_dense_2d() -> tensor<2x2xf32> {
  %0 = colgm.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// dense splat (rank-2)
func.func @const_dense_splat() -> tensor<2x2xf32> {
  %0 = colgm.constant dense<0.0> : tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
