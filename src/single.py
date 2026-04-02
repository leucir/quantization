import numpy as np

# Original value
x = 0.7823

# Configuration
bits = 4
x_min, x_max = -1.0, 1.0

# Step size (Δ)
n_levels = 2**bits - 1
delta = (x_max - x_min) / n_levels

# Quantize: map to grid, round, clip to valid range
q = np.clip(np.floor(x / delta + 0.5), x_min / delta, x_max / delta)

# Dequantize: back to value space
x_hat = q * delta

# The error
error = x - x_hat

print(f"Original:     {x}")
print(f"Δ (step size): {delta:.4f}")
print(f"Dequantized:  {x_hat:.4f}")
print(f"Error:        {error:.4f}")