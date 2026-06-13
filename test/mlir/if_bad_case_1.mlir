// verify error: condition must be i1 (got f32)
func.func @bad_cond_f32(%cond: f32) {
  colgm.if %cond {
    colgm.yield
  }
  func.return
}
