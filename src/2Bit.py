import numpy as np

w = np.array([0.3])
b = -0.15

# 2-bit quantization of w: range [0.3, 0.3] is a single point,
# so let's use a realistic range as if w came from a layer
w_min, w_max = -1.0, 1.0
n_levels = 2**2 - 1  # 3 levels → 4 values: -1.0, -0.33, 0.33, 1.0
delta = (w_max - w_min) / n_levels  # 0.6667

q = np.floor((w - w_min) / delta + 0.5)
q = np.clip(q, 0, n_levels)
w_q = q * delta + w_min  # snaps to 0.3333

# Decision boundary shifts
boundary_orig = -b / w[0]          # 0.5
boundary_quant = -b / w_q[0]       # 0.45

print(f"Original weight:  {w[0]:.4f}  →  boundary at x = {boundary_orig:.4f}")
print(f"Quantized weight: {w_q[0]:.4f}  →  boundary at x = {boundary_quant:.4f}")
print(f"\nInput x = 0.47:")
print(f"  Original model:  class {int(w[0] * 0.47 + b > 0)}")
print(f"  Quantized model: class {int(w_q[0] * 0.47 + b > 0)}")