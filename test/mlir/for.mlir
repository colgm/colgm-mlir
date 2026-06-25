// colgm dialect test: for op — positive cases
// parse -> print -> verify round-trip

// simple for loop (statement)
func.func @for_simple(%lb: index, %ub: index) {
  colgm.for %i = %lb to %ub {
    colgm.yield
  }
  func.return
}

// for loop with body ops (statement)
func.func @for_with_body(%lb: index, %ub: index, %a: tensor<f32>, %b: tensor<f32>) {
  colgm.for %i = %lb to %ub {
    %0 = colgm.add %a, %b : tensor<f32>
    colgm.yield
  }
  func.return
}

// for with single result (iter_args)
func.func @for_with_result(%lb: index, %ub: index, %a: tensor<f32>, %init: tensor<f32>) -> tensor<f32> {
  %0 = colgm.for %i = %lb to %ub iter_args(%acc = %init) -> tensor<f32> {
    colgm.yield %a : tensor<f32>
  }
  func.return %0 : tensor<f32>
}

// for with multi result (iter_args)
func.func @for_multi_result(%lb: index, %ub: index, %a: tensor<f32>, %b: tensor<f64>, %init_a: tensor<f32>, %init_b: tensor<f64>) -> tensor<f32> {
  %0, %1 = colgm.for %i = %lb to %ub iter_args(%acc_a = %init_a, %acc_b = %init_b) -> (tensor<f32>, tensor<f64>) {
    colgm.yield %a, %b : tensor<f32>, tensor<f64>
  }
  func.return %0 : tensor<f32>
}
