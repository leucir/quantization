import numpy as np

def quantize_dequantize(x, bits):
    x_min, x_max = x.min(), x.max()
    n_levels = 2**bits - 1
    delta = (x_max - x_min) / n_levels

    # Shift to zero-based range, quantize, shift back
    q = np.floor((x - x_min) / delta + 0.5)
    q = np.clip(q, 0, n_levels)
    x_hat = q * delta + x_min

    return x_hat

# Simulate a small layer of model weights
np.random.seed(42)
weights = np.random.randn(1000).astype(np.float32) * 0.5

print("Bit-width   Mean Error   Max Error    Error Std")
print("─" * 52)

for bits in [8, 4, 2]:
    x_hat = quantize_dequantize(weights, bits)
    errors = np.abs(weights - x_hat)
    print(f"  {bits:>2}         {errors.mean():.6f}    {errors.max():.6f}    {errors.std():.6f}")