// verify error: for yield count mismatch with result
func.func @bad_for_yield_mismatch(%lb: index, %ub: index) -> tensor<f32> {
  %0 = colgm.for %i = %lb to %ub -> tensor<f32> {
    colgm.yield
  }
  func.return %0 : tensor<f32>
}
