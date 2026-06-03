// verify error: duplicate axis
func.func @test_reduce_sum_bad(%arg: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
    %0 = colgm.reduce_sum %arg {axes = [1, 1]} : tensor<2x3x4xf32> -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
}
