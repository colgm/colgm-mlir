// colgm dialect test: slice op — bad case (wrong result shape)
// parse -> print -> verify should fail

func.func @bad_shape(%arg: tensor<3x4xf32>, %idx: index) -> tensor<4xf32> {
  %0 = colgm.slice %arg, %idx {axis = 1} : tensor<3x4xf32>, index -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
