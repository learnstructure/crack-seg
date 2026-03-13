import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Create large matrices
x = torch.randn(5000, 5000).to(device)
y = torch.randn(5000, 5000).to(device)

start = time.time()

z = torch.matmul(x, y)

torch.cuda.synchronize() if device.type == "cuda" else None

print("Time:", time.time() - start)