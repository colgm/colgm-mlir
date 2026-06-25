// verify error: for yield count mismatch with iter_args
func.func @bad_for_yield_mismatch(%lb: index, %ub: index, %init: tensor<f32>) -> tensor<f32> {
  %0 = colgm.for %i = %lb to %ub iter_args(%acc = %init) -> tensor<f32> {
    colgm.yield
  }
  func.return %0 : tensor<f32>
}
