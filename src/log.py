"""
Non-linear (logarithmic) quantization vs uniform quantization.

Uniform quantization spaces levels evenly across the value range.
This works well when values are roughly the same magnitude, but wastes
precision on small values when the range spans several orders of magnitude
(e.g. 0.001 to 1.0) — the fixed step size (~0.067 for 4-bit) is larger
than the smallest values themselves.

Logarithmic quantization applies uniform quantization in log-space instead.
Because log compresses large values and stretches small ones, the resulting
levels are geometrically spaced: dense near zero and sparse near the top.
This gives small values proportionally the same relative precision as large ones.

Run this script to see the absolute error of each scheme on a set of values
that spans three orders of magnitude.
"""

import math

BITS = 4
LEVELS = (1 << BITS) - 1  # 2^4 - 1 = 15 quantization levels


def uniform_quantize(v, lo, hi):
    """Quantize v into one of LEVELS evenly-spaced buckets between lo and hi."""
    step = (hi - lo) / LEVELS
    return lo + round((max(lo, min(hi, v)) - lo) / step) * step


def log_quantize(v, lo, hi):
    """Quantize v into one of LEVELS geometrically-spaced buckets between lo and hi.

    Maps v → log(v), quantizes uniformly in log-space, then maps back with exp.
    Requires lo > 0.
    """
    log_lo, log_hi = math.log(lo), math.log(hi)
    step = (log_hi - log_lo) / LEVELS
    bucket = round((max(log_lo, min(log_hi, math.log(v))) - log_lo) / step)
    return math.exp(log_lo + bucket * step)


# Values spanning three orders of magnitude (0.001 → 1.0)
values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
lo, hi = values[0], values[-1]

print(f"{'Value':>6}  {'Uniform err':>11}  {'Log err':>8}")
print("-" * 32)
for v in values:
    ue = abs(v - uniform_quantize(v, lo, hi))
    le = abs(v - log_quantize(v, lo, hi))
    print(f"{v:>6.3f}  {ue:>11.6f}  {le:>8.6f}")