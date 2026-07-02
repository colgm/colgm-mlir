func.func @test(%arg: tensor<i64>) -> i64 {
  %0 = colgm.cast %arg : tensor<i64> -> i64
  return %0 : i64
}