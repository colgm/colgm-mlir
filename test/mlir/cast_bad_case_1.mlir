// colgm dialect test: cast op — bad case (non-rank-0 tensor → scalar)
// parse -> print -> verify should fail: only rank-0 tensor can be cast to scalar

func.func @bad_non_scalar_tensor_to_scalar(%arg: tensor<2xi1>) -> i1 {
  %0 = colgm.cast %arg : tensor<2xi1> -> i1
  return %0 : i1
}
