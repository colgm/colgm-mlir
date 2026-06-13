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
