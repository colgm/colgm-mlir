// verify error: missing target_shape attribute
func.func @test_broadcast_bad(%arg: tensor<1x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = colgm.broadcast %arg : tensor<1x3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}
