// verify error: condition must be i1 (got i64)
func.func @bad_cond_i64(%cond: i64) {
  colgm.if %cond {
    colgm.yield
  }
  func.return
}
