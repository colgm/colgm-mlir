// verify error: element type mismatch
func.func @test_broadcast_bad(%arg: tensor<1x3x4xf32>) -> tensor<2x3x4xi32> {
    %0 = colgm.broadcast %arg {target_shape = [2, 3, 4]} : tensor<1x3x4xf32> -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
}
