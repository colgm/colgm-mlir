// colgm dialect test: exp op — bad case (i32 input)
// exp requires floating-point input; integer input should fail verify

func.func @bad_exp(%arg: tensor<i32>) -> tensor<i32> {
  %0 = colgm.exp %arg : tensor<i32>
  return %0 : tensor<i32>
}
