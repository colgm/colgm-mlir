
// tensor<i64> -> i64
func.func @test_0(%arg: tensor<i64>) -> i64 {
  %0 = colgm.cast %arg : tensor<i64> -> i64
  return %0 : i64
}

// i64 -> tensor<i64>
func.func @test_1(%arg: i64) -> tensor<i64> {
  %0 = colgm.cast %arg : i64 -> tensor<i64>
  return %0 : tensor<i64>
}

// tensor<i64> -> f64
func.func @test_2(%arg: tensor<i64>) -> f64 {
  %0 = colgm.cast %arg : tensor<i64> -> f64
  return %0 : f64
}

// tensor<f64> -> i64
func.func @test_3(%arg: tensor<f64>) -> i64 {
  %0 = colgm.cast %arg : tensor<f64> -> i64
  return %0 : i64
}

// tensor<i64> -> tensor<f64>
func.func @test_4(%arg: tensor<i64>) -> tensor<f64> {
  %0 = colgm.cast %arg : tensor<i64> -> tensor<f64>
  return %0 : tensor<f64>
}

// tensor<i64> -> index
func.func @test_5(%arg: tensor<i64>) -> index {
  %0 = colgm.cast %arg : tensor<i64> -> index
  return %0 : index
}

// index -> tensor<i64>
func.func @test_6(%arg: index) -> tensor<i64> {
  %0 = colgm.cast %arg : index -> tensor<i64>
  return %0 : tensor<i64>
}