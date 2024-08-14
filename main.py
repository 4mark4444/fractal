import numpy as np
import matplotlib.pyplot as plt

def douady_rabbit(height, width, max_iterations):
    y, x = np.ogrid[-1.5:1.5:height*1j, -2:1:width*1j]
    c = -0.123 + 0.745j
    z = x + y*1j

    divtime = max_iterations + np.zeros(z.shape, dtype=int)

    for i in range(max_iterations):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iterations)
        divtime[div_now] = i
        z[diverge] = 2

    return divtime

def main():
    height, width = 1000, 1500
    max_iterations = 100

    fractal = douady_rabbit(height, width, max_iterations)

    plt.figure(figsize=(12, 8))
    plt.imshow(fractal, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title("Douady Rabbit Fractal")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")

    plt.colorbar(label='Iteration count')
    plt.tight_layout()
    plt.savefig('douady_rabbit.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()