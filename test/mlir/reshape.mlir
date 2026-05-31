// colgm dialect test: reshape op
// parse -> print -> verify should all pass

func.func @test_reshape(%arg: tensor<2x3xf32>) -> tensor<6xf32> {
    %0 = colgm.reshape %arg {target_shape = [6]} : tensor<2x3xf32> -> tensor<6xf32>
    return %0 : tensor<6xf32>
}

func.func @test_reshape_2(%arg: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = colgm.reshape %arg {target_shape = [3, 2]} : tensor<2x3xf32> -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
}
