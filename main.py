import torch
import numpy as np
import matplotlib.pyplot as plt

def douady_rabbit_pytorch(height, width, max_iterations):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    y, x = torch.meshgrid(torch.linspace(-1.5, 1.5, height), torch.linspace(-2, 1, width))
    c = torch.tensor(-0.123 + 0.745j, device=device)
    z = torch.complex(x, y).to(device)

    divtime = torch.full(z.shape, max_iterations, dtype=torch.int32, device=device)

    for i in range(max_iterations):
        z = z**2 + c
        # magic |
        diverge = torch.abs(z) > 2
        # magic |

        # REF: generated with the help from claude.
        div_now = diverge & (divtime == max_iterations)
        divtime[div_now] = i
        z[diverge] = 2

    return divtime.cpu().numpy()

def main():
    height, width = 2000, 3000  # Increased resolution
    max_iterations = 500  # Increased iterations for more detail

    fractal = douady_rabbit_pytorch(height, width, max_iterations)

    plt.figure(figsize=(16, 10))
    plt.imshow(fractal, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title("Douady Rabbit Fractal (PyTorch Parallel Computation)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")

    plt.colorbar(label='Iteration count')
    plt.tight_layout()
    plt.savefig('douady_rabbit_pytorch.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()