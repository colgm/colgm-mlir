// colgm dialect test: for op — positive cases
// parse -> print -> verify round-trip

// simple for loop
func.func @for_simple(%lb: index, %ub: index) {
  colgm.for %i = %lb to %ub {
    colgm.yield
  }
  func.return
}

// for loop with body ops
func.func @for_with_body(%lb: index, %ub: index, %a: tensor<f32>, %b: tensor<f32>) {
  colgm.for %i = %lb to %ub {
    %0 = colgm.add %a, %b : tensor<f32>
    colgm.yield
  }
  func.return
}
