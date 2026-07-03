// colgm dialect test: cast op — bad case (scalar → non-rank-0 tensor)
// parse -> print -> verify should fail: scalar can only be cast to rank-0 tensor

func.func @bad_scalar_to_non_scalar_tensor(%arg: i1) -> tensor<2xi1> {
  %0 = colgm.cast %arg : i1 -> tensor<2xi1>
  return %0 : tensor<2xi1>
}
