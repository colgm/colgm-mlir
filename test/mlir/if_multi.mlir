// colgm dialect test: if op — multi-result positive cases
// parse -> print -> verify round-trip

// if with two results
func.func @if_two_results(%cond: i1, %a: tensor<f32>, %b: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = colgm.if %cond -> (tensor<f32>, tensor<f32>) {
    colgm.yield %a, %b : tensor<f32>, tensor<f32>
  } else {
    colgm.yield %b, %a : tensor<f32>, tensor<f32>
  }
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// if with three results, computed in body
func.func @if_three_results(%cond: i1, %x: tensor<f32>, %y: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
  %0:3 = colgm.if %cond -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %1 = colgm.add %x, %y : tensor<f32>
    %2 = colgm.sub %x, %y : tensor<f32>
    %3 = colgm.mul %x, %y : tensor<f32>
    colgm.yield %1, %2, %3 : tensor<f32>, tensor<f32>, tensor<f32>
  } else {
    colgm.yield %x, %y, %x : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.return %0#0, %0#1, %0#2 : tensor<f32>, tensor<f32>, tensor<f32>
}

// multi-result with mixed types
func.func @if_mixed_types(%cond: i1) -> (tensor<f32>, tensor<i32>) {
  %a = colgm.constant dense<1.0> : tensor<f32>
  %b = colgm.constant dense<42> : tensor<i32>
  %c = colgm.constant dense<3.0> : tensor<f32>
  %d = colgm.constant dense<99> : tensor<i32>
  %0:2 = colgm.if %cond -> (tensor<f32>, tensor<i32>) {
    colgm.yield %a, %b : tensor<f32>, tensor<i32>
  } else {
    colgm.yield %c, %d : tensor<f32>, tensor<i32>
  }
  func.return %0#0, %0#1 : tensor<f32>, tensor<i32>
}
