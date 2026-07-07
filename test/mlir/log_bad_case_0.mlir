// colgm dialect test: log op — bad case (i64 input)
// log requires floating-point input; integer input should fail verify

func.func @bad_log(%arg: tensor<i64>) -> tensor<i64> {
  %0 = colgm.log %arg : tensor<i64>
  return %0 : tensor<i64>
}
