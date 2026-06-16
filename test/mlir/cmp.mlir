// colgm dialect test: comparison ops (eq, ne, lt, le, gt, ge)
// parse -> print -> verify should all pass

// scalar comparison (rank-0 tensors)
func.func @cmp_eq_scalar(%a: tensor<f32>, %b: tensor<f32>) -> tensor<i1> {
  %0 = colgm.cmp_eq %a, %b : tensor<f32> -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @cmp_ne_scalar(%a: tensor<f32>, %b: tensor<f32>) -> tensor<i1> {
  %0 = colgm.cmp_ne %a, %b : tensor<f32> -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @cmp_lt_scalar(%a: tensor<f32>, %b: tensor<f32>) -> tensor<i1> {
  %0 = colgm.cmp_lt %a, %b : tensor<f32> -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @cmp_le_scalar(%a: tensor<f32>, %b: tensor<f32>) -> tensor<i1> {
  %0 = colgm.cmp_le %a, %b : tensor<f32> -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @cmp_gt_scalar(%a: tensor<f32>, %b: tensor<f32>) -> tensor<i1> {
  %0 = colgm.cmp_gt %a, %b : tensor<f32> -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @cmp_ge_scalar(%a: tensor<f32>, %b: tensor<f32>) -> tensor<i1> {
  %0 = colgm.cmp_ge %a, %b : tensor<f32> -> tensor<i1>
  func.return %0 : tensor<i1>
}

// tensor comparison (rank-2)
func.func @cmp_eq_tensor(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xi1> {
  %0 = colgm.cmp_eq %x, %y : tensor<2x3xf32> -> tensor<2x3xi1>
  func.return %0 : tensor<2x3xi1>
}

func.func @cmp_lt_tensor(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xi1> {
  %0 = colgm.cmp_lt %x, %y : tensor<2x3xf32> -> tensor<2x3xi1>
  func.return %0 : tensor<2x3xi1>
}

// i32 comparison
func.func @cmp_eq_i32(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i1> {
  %0 = colgm.cmp_eq %a, %b : tensor<i32> -> tensor<i1>
  func.return %0 : tensor<i1>
}
