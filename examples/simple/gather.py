import slope


print('#1')
x = slope.tensor([[0.,1.],[2.,3.]], dtype=slope.float32)
w = slope.tensor([[0,0],[1,1]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather_nd(w)
print(f"{y=}")

# print('\n#2')
# x = slope.tensor([[0.,1.],[2.,3.]], dtype=slope.float32)
# w = slope.tensor([[1],[0]]).cast(slope.int64)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather_nd(w)
# print(f"{y=}")

# print('\n#3')
# x = slope.tensor([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=slope.float32)
# w = slope.tensor([[0,1],[1,0]], dtype=slope.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather_nd(w)
# print(f"{y=}")


# print('\n#4')
# x = slope.tensor([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=slope.float32)
# w = slope.tensor([[[0,1]],[[1,0]]], dtype=slope.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather_nd(w)
# print(f"{y=}")

# print('\n#5')
# x = slope.tensor([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=slope.float32)
# w = slope.tensor([[1],[0]], dtype=slope.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather_nd(w, 1)
# print(f"{y=}")

'''

func.func @main (%x0: tensor<2x2xf32>, %x1: tensor<2x2xi32>) -> (tensor<2xf32>)
{
    %y0_ = "stablehlo.gather"(%x0, %x1) {
      dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
      slice_sizes = dense<[1, 1]> : tensor<2xi64>,
      indices_are_sorted = false
    }  : (tensor<2x2xf32>,tensor<2x2xi32>) -> tensor<2x1xf32>
    %y0 = "stablehlo.reshape"(%y0_)  : (tensor<2x1xf32>) -> tensor<2xf32>
    
    "func.return"(%y0): (tensor<2xf32>) -> ()
}
'''
'''

func.func @main (%x0: tensor<2x2xf32>, %x1: tensor<2x1xi32>) -> (tensor<2x2xf32>)
{
    %y0 = "stablehlo.gather"(%x0, %x1) {
      dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
      slice_sizes = dense<[1, 2]> : tensor<2xi64>,
      indices_are_sorted = false
    }  : (tensor<2x2xf32>,tensor<2x1xi32>) -> tensor<2x2xf32>
    
    "func.return"(%y0): (tensor<2x2xf32>) -> ()
}
'''

'''

func.func @main (%x0: tensor<2x2x2xf32>, %x1: tensor<2x2xi32>) -> (tensor<2x2xf32>)
{
    %y0 = "stablehlo.gather"(%x0, %x1) {
      dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
      slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>,
      indices_are_sorted = false
    }  : (tensor<2x2x2xf32>,tensor<2x2xi32>) -> tensor<2x2xf32>
    
    "func.return"(%y0): (tensor<2x2xf32>) -> ()
}
'''


'''
<stdin>:3:11: error: start_index_map size (2) 
is not equal to size of index dimension (1) of start_indices (1)
func.func @main (%x0: tensor<2x2x2xf32>, %x1: tensor<2x1x2xi32>) -> (tensor<2x2x2xf32>)
{
    %y0 = "stablehlo.gather"(%x0, %x1) {
      dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
      slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>,
      indices_are_sorted = false
    }  : (tensor<2x2x2xf32>,tensor<2x1x2xi32>) -> tensor<2x2x2xf32>
    
    "func.return"(%y0): (tensor<2x2x2xf32>) -> ()
}
'''


###############

# x = slope.arange(10, dtype=slope.float32).reshape(2,5)
# w = slope.tensor([1,0])[..., None]
# w = w.cast(slope.int64)
# y = x.gather_nd(w)
# print(f"{x=}")
# print(f"{w=}")
# print(f"{y=}")

# x = slope.arange(24, dtype=slope.float32).reshape(4,3,2)
# w = x
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather_nd(w)
# breakpoint()
# print(f"{y=}")

#######################

# x = slope.ones(8)
# print(f"before: {x=}")
# w = slope.tensor([[4], [3], [1], [7]], dtype=slope.int32)
# u = slope.tensor([9., 10., 11., 12.])
# y = slope.scatter_nd(x,w,u)

# print(f"{w=}")
# print(f"{u=}")
# print(f"{x=}")
# print(f"{y=}")