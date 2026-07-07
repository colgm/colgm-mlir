// colgm dialect test: tanh op — bad case (i32 input)
// tanh requires floating-point input; integer input should fail verify

func.func @bad_tanh(%arg: tensor<i32>) -> tensor<i32> {
  %0 = colgm.tanh %arg : tensor<i32>
  return %0 : tensor<i32>
}
