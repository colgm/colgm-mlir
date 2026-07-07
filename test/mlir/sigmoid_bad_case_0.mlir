// colgm dialect test: sigmoid op — bad case (i32 input)
// sigmoid requires floating-point input; integer input should fail verify

func.func @bad_sigmoid(%arg: tensor<i32>) -> tensor<i32> {
  %0 = colgm.sigmoid %arg : tensor<i32>
  return %0 : tensor<i32>
}
