// colgm dialect test: constfold pass — elements with all-constant operands
// run: colgm-opt --colgm-const-fold

func.func @fold_elements_1d() -> tensor<3xi64> {
  %c0 = colgm.constant dense<1> : tensor<i64>
  %c1 = colgm.constant dense<2> : tensor<i64>
  %c2 = colgm.constant dense<3> : tensor<i64>
  %r = colgm.elements(%c0, %c1, %c2) {target_shape = [3]} : tensor<i64> -> tensor<3xi64>
  func.return %r : tensor<3xi64>
}

func.func @fold_elements_2d() -> tensor<2x2xf32> {
  %c0 = colgm.constant dense<1.0> : tensor<f32>
  %c1 = colgm.constant dense<2.0> : tensor<f32>
  %c2 = colgm.constant dense<3.0> : tensor<f32>
  %c3 = colgm.constant dense<4.0> : tensor<f32>
  %r = colgm.elements(%c0, %c1, %c2, %c3) {target_shape = [2, 2]} : tensor<f32> -> tensor<2x2xf32>
  func.return %r : tensor<2x2xf32>
}

// elements with mixed (non-constant) operands — should NOT be folded
func.func @no_fold_mixed(%a: tensor<f64>) -> tensor<3xf64> {
  %c0 = colgm.constant dense<1.0> : tensor<f64>
  %c1 = colgm.constant dense<2.0> : tensor<f64>
  %r = colgm.elements(%c0, %c1, %a) {target_shape = [3]} : tensor<f64> -> tensor<3xf64>
  func.return %r : tensor<3xf64>
}
