// colgm dialect test: reduce_max op — bad case (missing axes)
// reduce_max requires axes attribute

func.func @bad_reduce_max(%arg: tensor<3x4xf64>) -> tensor<4xf64> {
  %0 = colgm.reduce_max %arg : tensor<3x4xf64> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}
