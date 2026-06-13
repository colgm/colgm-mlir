// verify error: condition must be i1 (got tensor)
func.func @bad_cond_tensor(%cond: tensor<i1>) {
  colgm.if %cond {
    colgm.yield
  }
  func.return
}
