import numpy as np

# Create a smooth range of values
x = np.linspace(-1.0, 1.0, 200)

bits = 4
x_min, x_max = -1.0, 1.0
n_levels = 2**bits - 1
delta = (x_max - x_min) / n_levels

q = np.floor((x - x_min) / delta + 0.5)
q = np.clip(q, 0, n_levels)
x_hat = q * delta + x_min
errors = x - x_hat

# Print a text-based visualization
print(f"Error pattern at {bits}-bit quantization (Δ = {delta:.4f})")
print(f"{'Value':>8}  {'Error':>8}  Pattern")
print("─" * 45)

for i in range(0, len(x), 10):
    bar_len = int(abs(errors[i]) / delta * 40)
    direction = "+" if errors[i] >= 0 else "-"
    bar = direction * bar_len
    print(f"{x[i]:>8.4f}  {errors[i]:>8.4f}  {bar}")