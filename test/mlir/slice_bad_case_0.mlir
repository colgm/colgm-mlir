// colgm dialect test: slice op — bad case (axis out of range)
// parse -> print -> verify should fail

func.func @bad_axis(%arg: tensor<3x4xf32>, %idx: index) -> tensor<3xf32> {
  %0 = colgm.slice %arg, %idx {axis = 2} : tensor<3x4xf32>, index -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
