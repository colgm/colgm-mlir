// colgm dialect test: if op — positive cases
// parse -> print -> verify round-trip

// if with then and else
func.func @if_then_else(%cond: i1) {
  colgm.if %cond {
    colgm.yield
  } else {
    colgm.yield
  }
  func.return
}

// if with then only (no else)
func.func @if_then_only(%cond: i1) {
  colgm.if %cond {
    colgm.yield
  }
  func.return
}

// if with body ops
func.func @if_with_body(%cond: i1, %a: tensor<f32>, %b: tensor<f32>) {
  colgm.if %cond {
    %0 = colgm.add %a, %b : tensor<f32>
    colgm.yield
  } else {
    %1 = colgm.sub %a, %b : tensor<f32>
    colgm.yield
  }
  func.return
}

// nested if
func.func @if_nested(%x: i1, %y: i1) {
  colgm.if %x {
    colgm.if %y {
      colgm.yield
    } else {
      colgm.yield
    }
    colgm.yield
  }
  func.return
}

// if with result (then/else)
func.func @if_with_result(%cond: i1, %a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
  %0 = colgm.if %cond -> tensor<f32> {
    colgm.yield %a : tensor<f32>
  } else {
    colgm.yield %b : tensor<f32>
  }
  func.return %0 : tensor<f32>
}

// if with result, nested blocks
func.func @if_result_with_ops(%cond: i1, %x: tensor<f32>, %y: tensor<f32>) -> tensor<f32> {
  %0 = colgm.if %cond -> tensor<f32> {
    %1 = colgm.add %x, %y : tensor<f32>
    colgm.yield %1 : tensor<f32>
  } else {
    %2 = colgm.sub %x, %y : tensor<f32>
    colgm.yield %2 : tensor<f32>
  }
  func.return %0 : tensor<f32>
}
