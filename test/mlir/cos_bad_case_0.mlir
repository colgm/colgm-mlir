// colgm dialect test: cos op — bad case (i32 input)
// cos requires floating-point input; integer input should fail verify

func.func @bad_cos(%arg: tensor<i32>) -> tensor<i32> {
  %0 = colgm.cos %arg : tensor<i32>
  return %0 : tensor<i32>
}
