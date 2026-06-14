// colgm dialect test: slice op — bad case (missing axis attr)
// parse -> print -> verify should fail

func.func @bad_missing_axis(%arg: tensor<3x4xf32>, %idx: index) -> tensor<4xf32> {
  %0 = colgm.slice %arg, %idx : tensor<3x4xf32>, index -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
