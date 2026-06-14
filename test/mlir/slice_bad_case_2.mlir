// colgm dialect test: slice op — bad case (index type wrong)
// parse -> print -> verify should fail: index must be index type

func.func @bad_index_type(%arg: tensor<3x4xf32>, %idx: i64) -> tensor<4xf32> {
  %0 = colgm.slice %arg, %idx {axis = 0} : tensor<3x4xf32>, i64 -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
