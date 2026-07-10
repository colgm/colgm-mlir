// colgm dialect test: print op — positive cases
// parse -> print -> verify round-trip

// print single operand (float)
func.func @print_single(%x: tensor<f32>) {
  colgm.print %x : tensor<f32>
  return
}

// print multiple operands (mixed types)
func.func @print_multi(%a: tensor<f32>, %b: tensor<i64>, %c: tensor<f64>) {
  colgm.print %a, %b, %c : tensor<f32>, tensor<i64>, tensor<f64>
  return
}

// print two operands of same type
func.func @print_two(%x: tensor<f32>, %y: tensor<f32>) {
  colgm.print %x, %y : tensor<f32>, tensor<f32>
  return
}

// print int tensor
func.func @print_int(%x: tensor<i32>) {
  colgm.print %x : tensor<i32>
  return
}

// print with no operands (empty print)
func.func @print_empty() {
  colgm.print
  return
}
