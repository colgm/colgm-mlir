// verify error: lower bound must be index (got i64)
func.func @bad_lb_i64(%lb: i64, %ub: index) {
  colgm.for %i = %lb to %ub {
    colgm.yield
  }
  func.return
}
