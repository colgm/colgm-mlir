// verify error: input rank > output rank
func.func @test_broadcast_bad(%arg: tensor<2x3x4xf32>) -> tensor<3x4xf32> {
    %0 = colgm.broadcast %arg {target_shape = [3, 4]} : tensor<2x3x4xf32> -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
}
