// verify error: upper bound must be index (got f32)
func.func @bad_ub_f32(%lb: index, %ub: f32) {
  colgm.for %i = %lb to %ub {
    colgm.yield
  }
  func.return
}
