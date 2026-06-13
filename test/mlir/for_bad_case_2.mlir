// verify error: both bounds wrong type
func.func @bad_both_i32(%lb: i32, %ub: i32) {
  colgm.for %i = %lb to %ub {
    colgm.yield
  }
  func.return
}
