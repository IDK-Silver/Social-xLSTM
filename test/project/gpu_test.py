#!/usr/bin/env python3
import torch

def main():
    # Check if CUDA (GPU support) is available
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    # Print CUDA device information
    print("CUDA is available!")
    print("Number of CUDA devices:", torch.cuda.device_count())
    current_device = torch.cuda.current_device()
    print("Current CUDA device index:", current_device)
    print("Current CUDA device name:", torch.cuda.get_device_name(current_device))

    # Set device to CUDA
    device = torch.device("cuda")

    # Create two random tensors on the GPU
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    print("\nTensor x on GPU:")
    print(x)
    print("\nTensor y on GPU:")
    print(y)

    # Perform a simple matrix multiplication on the GPU
    z = torch.matmul(x, y)
    print("\nResult of matrix multiplication (x @ y):")
    print(z)

    # Optionally, move the result back to the CPU and convert to a numpy array for further processing
    z_cpu = z.cpu().numpy()
    print("\nResult transferred to CPU (as numpy array):")
    print(z_cpu)

if __name__ == "__main__":
    main()