// colgm dialect test: transpose op
// parse -> print -> verify should all pass

func.func @test_transpose_2d(%arg: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = colgm.transpose %arg {permutation = [1, 0]} : tensor<2x3xf32> -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
}

func.func @test_transpose_3d_identity(%arg: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = colgm.transpose %arg {permutation = [0, 1, 2]} : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
}

func.func @test_transpose_scalar(%arg: tensor<f32>) -> tensor<f32> {
    %0 = colgm.transpose %arg {permutation = []} : tensor<f32> -> tensor<f32>
    return %0 : tensor<f32>
}
