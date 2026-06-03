// colgm dialect test: reduce_sum op
// parse -> print -> verify should all pass

func.func @test_reduce_mid_dim(%arg: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
    %0 = colgm.reduce_sum %arg {axes = [1]} : tensor<2x3x4xf32> -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
}

func.func @test_reduce_two_dims(%arg: tensor<2x3x4xf32>) -> tensor<3xf32> {
    %0 = colgm.reduce_sum %arg {axes = [0, 2]} : tensor<2x3x4xf32> -> tensor<3xf32>
    return %0 : tensor<3xf32>
}

func.func @test_reduce_all(%arg: tensor<2x3xf32>) -> tensor<f32> {
    %0 = colgm.reduce_sum %arg {axes = [0, 1]} : tensor<2x3xf32> -> tensor<f32>
    return %0 : tensor<f32>
}

func.func @test_reduce_empty_axes(%arg: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = colgm.reduce_sum %arg {axes = []} : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}

func.func @test_reduce_scalar(%arg: tensor<f32>) -> tensor<f32> {
    %0 = colgm.reduce_sum %arg {axes = []} : tensor<f32> -> tensor<f32>
    return %0 : tensor<f32>
}

func.func @test_reduce_first_dim(%arg: tensor<4x5xf32>) -> tensor<5xf32> {
    %0 = colgm.reduce_sum %arg {axes = [0]} : tensor<4x5xf32> -> tensor<5xf32>
    return %0 : tensor<5xf32>
}
