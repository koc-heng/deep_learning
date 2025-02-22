import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


print("=== Task 1 ===")
tensor1 = torch.tensor([[2, 3, 1],
                        [5, -2, 1]])
print("Tensor1:", tensor1)
print("Shape of Tensor1:", tensor1.shape)
print("Dtype of Tensor1:", tensor1.dtype)

print("=== Task 2 ===")
tensor2 = torch.rand((3, 4, 2))  
print("Tensor2:", tensor2)
print("Shape of Tensor2:", tensor2.shape)
print("Dtype of Tensor2:", tensor2.dtype)

print("=== Task 3 ===")
tensor3 = torch.ones((2, 1, 5))  
print("Tensor3:", tensor3)
print("Shape of Tensor3:", tensor3.shape)
print("Dtype of Tensor3:", tensor3.dtype)

print("=== Task 4 ===")
A = torch.tensor([[1, 2, 4],
                  [2, 1, 3]])
B = torch.tensor([[5],
                  [2],
                  [1]])
C = A @ B 
print("Tensor A:", A)
print("Tensor B:", B)
print("Result of A x B:", C)
print("Shape of the result:", C.shape)

print("=== Task 5 ===")
A = torch.tensor([[1, 2],
                  [2, 3],
                  [-1, 3]])
B = torch.tensor([[5, 4],
                  [2, 1],
                  [1,-5]])
                  
C = A * B 
print("Tensor A:", A)
print("Tensor B:", B)
print("Result of A x B:", C)
print("Shape of the result:", C.shape)