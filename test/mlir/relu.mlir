// colgm dialect test: relu op
// parse -> print -> verify should all pass

func.func @main(%arg0: tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
  %0 = colgm.relu %arg0 : tensor<1x3x4xf32>
  return %0 : tensor<1x3x4xf32>
}
