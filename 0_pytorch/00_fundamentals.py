import torch

# print(torch.__version__)
torch.manual_seed(42)

### 1. Tensor
# scalar
scalar = torch.tensor(9)
print(f"scalar - shape={scalar.size()}, dim={scalar.ndim}, item={scalar.item()}")
print()

# vector
vector = torch.tensor([6, 9])
print(f"vector - shape={vector.size()}, dim={vector.ndim}")

# matrix
matrix = torch.tensor([[1, 2, 3], [3, 2, 1]])
print(f"matrix - shape={matrix.size()}, dim={matrix.ndim}")

# tensor
tensor = torch.tensor([[[1, 2, 3], [1, 2, 3]]])
print(f"tensor - shape={tensor.size()}, dim={tensor.ndim}")

# random tensor

random_tensor = torch.rand(size=(2, 3))
print("random_tensor: ", random_tensor)
print(f"random_tensor, shape={random_tensor.size()}, type={random_tensor.dtype}")

# tensor ones
ones = torch.ones(size=(2, 3))
print("ones: ", ones)

# zeros tensor
zeros = torch.zeros(size=(4, 2))
print("zeros: ", zeros)

# create tensor arrange
print(torch.arange(0, 10, 1))

# get information of tensor
tensor_float32 = torch.rand(size=(3, 2), 
							dtype=torch.float16, 
							device="cpu")
print(tensor_float32)
print(f"information of tensor_float32- dtype={tensor_float32.dtype}, device={tensor_float32.device}")


### 2. tensor operations
print("=======2. tensor operations=======")

a = torch.tensor([[1, 2, 3], [1, 2, 3]])
b = torch.tensor([[2, 1, 4], [1, 2, -1]])
print("Tensor a: ", a.size())
print("Tensor b: ", b.size())

# Element-wise multiplication
print("Element-wise multiplication: ", a*b)

# print("Inner product (dot-product): ", torch.dot(a, b))
# Note that inner product or dot-product only for 2 vectors have same dimention
print("Matrix multiply: ", a@b.T)
print("Matrix multiply: ",torch.matmul(a, b.T))





