// colgm dialect test: slice op
// parse -> print -> verify should all pass

func.func @slice_axis0(%arg: tensor<3x4xf32>, %idx: index) -> tensor<4xf32> {
  %0 = colgm.slice %arg, %idx {axis = 0} : tensor<3x4xf32>, index -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @slice_axis1(%arg: tensor<2x3x4xf64>, %idx: index) -> tensor<2x4xf64> {
  %0 = colgm.slice %arg, %idx {axis = 1} : tensor<2x3x4xf64>, index -> tensor<2x4xf64>
  return %0 : tensor<2x4xf64>
}
