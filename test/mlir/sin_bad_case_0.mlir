// colgm dialect test: sin op — bad case (i32 input)
// sin requires floating-point input; integer input should fail verify

func.func @bad_sin(%arg: tensor<i32>) -> tensor<i32> {
  %0 = colgm.sin %arg : tensor<i32>
  return %0 : tensor<i32>
}
