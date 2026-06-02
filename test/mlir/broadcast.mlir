// colgm dialect test: broadcast op
// parse -> print -> verify should all pass

func.func @test_broadcast_expand_dim(%arg: tensor<1x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = colgm.broadcast %arg {target_shape = [2, 3, 4]} : tensor<1x3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}

func.func @test_broadcast_new_leading_dim(%arg: tensor<3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = colgm.broadcast %arg {target_shape = [2, 3, 4]} : tensor<3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}

func.func @test_broadcast_scalar(%arg: tensor<f32>) -> tensor<2x3xf32> {
    %0 = colgm.broadcast %arg {target_shape = [2, 3]} : tensor<f32> -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
}

func.func @test_broadcast_identity(%arg: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = colgm.broadcast %arg {target_shape = [2, 3, 4]} : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}
