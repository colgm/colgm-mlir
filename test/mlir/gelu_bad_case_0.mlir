// colgm dialect test: gelu op — bad case (i32 input)
// gelu requires floating-point input; integer input should fail verify

func.func @bad_gelu(%arg: tensor<i32>) -> tensor<i32> {
  %0 = colgm.gelu %arg : tensor<i32>
  return %0 : tensor<i32>
}
