// colgm dialect test: sqrt op — bad case (i32 input)
// sqrt requires floating-point input; integer input should fail verify

func.func @bad_sqrt(%arg: tensor<i32>) -> tensor<i32> {
  %0 = colgm.sqrt %arg : tensor<i32>
  return %0 : tensor<i32>
}
