import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())          # Should print True
print("GPU name:", torch.cuda.get_device_name(0))            # Should show RTX 5060 Ti

# Crucial test – actually run something on the GPU
try:
    x = torch.tensor([1.0, 2.0]).cuda()
    print("GPU tensor test successful:", x)
except Exception as e:
    print("GPU tensor test failed:", e)