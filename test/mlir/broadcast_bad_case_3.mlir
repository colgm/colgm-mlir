// verify error: output shape mismatch with target_shape
func.func @test_broadcast_bad(%arg: tensor<1x3x4xf32>) -> tensor<2x3x5xf32> {
    %0 = colgm.broadcast %arg {target_shape = [2, 3, 4]} : tensor<1x3x4xf32> -> tensor<2x3x5xf32>
    return %0 : tensor<2x3x5xf32>
}
