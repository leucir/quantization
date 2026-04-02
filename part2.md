# Intelligence Survives Compression — But Not Unchanged (Part 2)

## Under the Hood: Building Quantization From Scratch

---

In Part 1, we built the intuition. Quantization is lossy, dequantization reveals the gap, and the error is the price of compression. We saw the formula, the tradeoff triangle, and the modern methods that make it all practical.

Now let's open the hood.

This part is a workshop. We're going to take the formula from Part 1, break it into pieces, understand what each piece does, and then write code that actually runs. By the end, you'll be able to quantize a tensor yourself and know exactly what happened at every step.

---

## A Note on Scope

Quantization is a large and fast-moving research area. There are entire families of methods we haven't touched in this series — quantization-aware training, vector quantization, non-uniform schemes, binary and ternary networks, mixed-precision training, and more. New papers come out regularly. New techniques keep shifting what's possible.

This series doesn't try to cover all of it. We chose one path — uniform linear quantization — and went deep. Think of it as a foundation, not a survey. Once you understand the mechanics here, the more advanced methods are variations on themes you'll already recognize.

This is a starting point, not the whole map.

---

## The Formula, One Piece at a Time

Here it is again, the formula from Part 1:

```
Q(x) = Δ · floor(x / Δ + 0.5)
```

Five components. Each one doing something specific. Let's walk through them.

---

### Δ — The Step Size

Delta is the most important number in the whole operation. It determines how coarse or fine your quantization grid is.

The calculation:

```
Δ = (max - min) / (2^bits - 1)
```

You take the range of values you need to represent — from the smallest weight to the largest — and divide it into evenly spaced steps. The number of steps is determined by how many bits you have. More bits, more steps, smaller Δ.

Think of Δ as the resolution of a ruler. A ruler marked in centimeters has Δ = 1cm. One marked in millimeters has Δ = 0.1cm. Same ruler, different precision. The millimeter ruler can express more detail, but it needs more marks — more bits.

Here's what Δ looks like for the same value range (say, -1.0 to 1.0) at different bit-widths:

```
Bits    Levels    Δ (step size)
────────────────────────────────
8       255       0.0078
4       15        0.1333
2       3         0.6667
```

At 8 bits, you can distinguish values about 0.008 apart. At 2 bits, you only have four possible values to work with. Every weight in the model has to become one of those four values. That's where the loss comes from.

---

### The Division: x / Δ

This step takes your original value and asks: "where does this fall on the quantization grid?"

If your value is `0.45` and your step size is `0.1333`, then `0.45 / 0.1333 ≈ 3.375`. That means your value sits between the 3rd and 4th step on the grid.

It's like converting meters to ruler-marks. You're not measuring in the original units anymore — you're counting how many steps from zero.

---

### The Offset: + 0.5

This is a small detail that matters a lot. Adding 0.5 before flooring is what turns this into *rounding to nearest* instead of *always rounding down*.

Without it, `3.375` would floor to `3`. With it, `3.375 + 0.5 = 3.875`, which still floors to `3`. But if the value were `3.6`, then `3.6 + 0.5 = 4.1`, which floors to `4` — correctly rounding up to the nearer grid point.

Without this offset, every value would systematically round downward. Over billions of parameters, that bias adds up. The `+ 0.5` keeps the rounding fair.

---

### The Floor: floor(...)

This is where the information actually dies.

`floor(3.875)` gives you `3`. The `.875` is gone. Forever. No operation can recover it. The fractional position between grid points — which was the only thing distinguishing your value from its neighbors — has been erased.

Everything before the floor was reversible. Everything after is not. This one operation is the one-way door from Part 1.

---

### The Rescale: Δ · (...)

After flooring, you have an integer — a grid position. But the model doesn't work with grid positions. It works with weight values. So you multiply back by Δ to return to the original scale.

`3 × 0.1333 = 0.4`. That's your dequantized value. Your original was `0.45`. The error is `0.05`.

This last step is technically the dequantization. It's baked right into the formula. Quantize and dequantize in one expression.

---

## Let's Build It

Enough theory. Let's write code.

We'll start with a single value, then scale up to a tensor, then look at how the error behaves across bit-widths.

### Quantizing a Single Value

```python
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
```

```
Original:     0.7823
Δ (step size): 0.1333
Dequantized:  0.8000
Error:        -0.0177
```

Run this and you'll see the gap. Change `bits` from 4 to 8 to 2 and watch the error change. That's the tradeoff triangle from Part 1, in numbers.

### Quantizing a Tensor

A single value is instructive. A tensor is realistic. Model weights don't come alone — they come in layers of thousands or millions of values.

```python
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
```

```
Bit-width   Mean Error   Max Error    Error Std
────────────────────────────────────────────────────
   8         0.003464    0.006955    0.002030
   4         0.058760    0.118192    0.032935
   2         0.302061    0.590383    0.173573
```

The pattern is clear: fewer bits, more error. But notice the max error too — at 2 bits, the worst-case error is substantial. Somewhere in that tensor, a weight got moved far from where it should be. And that weight is part of a computation that feeds into every output the model produces.

### When Error Flips a Decision

What does that error actually *do*? Consider the simplest classifier: a single weight `w = 0.3` and a bias `b = -0.15`. The decision rule is `w * x + b > 0`, which means the boundary sits at `x = 0.5`. Anything above is class 1, anything below is class 0.

```python
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
```

```
Original weight:  0.3000  →  boundary at x = 0.5000
Quantized weight: 0.3333  →  boundary at x = 0.4500

Input x = 0.47:
  Original model:  class 0
  Quantized model: class 1
```

The weight moved from `0.3` to `0.33`. That's a small shift — but it pushed the decision boundary from `0.5` down to `0.45`. An input at `x = 0.47`, which was safely class 0, is now class 1. The model's answer changed, not because it learned something new, but because a weight got rounded.

This is one weight in one dimension. Real models have billions of weights and decision boundaries in spaces too large to visualize. But the mechanism is the same: quantization moves weights, moved weights shift boundaries, shifted boundaries change outputs. The inputs most at risk are the ones closest to the boundary — the cases the model was least certain about.

### Visualizing the Error

Let's look at what the error actually looks like across a range of values. This is where things start to get interesting — and where Part 3's security discussion will pick up.

```python
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

# '+' = original value was above its quantized grid point
# '-' = original value was below its quantized grid point
for i in range(0, len(x), 10):
    bar_len = int(abs(errors[i]) / delta * 40)
    direction = "+" if errors[i] >= 0 else "-"
    bar = direction * bar_len
    print(f"{x[i]:>8.4f}  {errors[i]:>8.4f}  {bar}")
```

```
Error pattern at 4-bit quantization (Δ = 0.1333)
   Value     Error  Pattern
─────────────────────────────────────────────
 -1.0000    0.0000  
 -0.8995   -0.0328  ---------
 -0.7990   -0.0657  -------------------
 -0.6985    0.0348  ++++++++++
 -0.5980    0.0020  
 -0.4975   -0.0308  ---------
 -0.3970   -0.0637  -------------------
 -0.2965    0.0369  +++++++++++
 -0.1960    0.0040  +
 -0.0955   -0.0288  --------
  0.0050   -0.0616  ------------------
  0.1055    0.0389  +++++++++++
  0.2060    0.0060  +
  0.3065   -0.0268  --------
  0.4070   -0.0596  -----------------
  0.5075    0.0409  ++++++++++++
  0.6080    0.0080  ++
  0.7085   -0.0248  -------
  0.8090   -0.0576  -----------------
  0.9095    0.0429  ++++++++++++
```

The error isn't random noise. It has a sawtooth pattern — rising smoothly between grid points, then snapping back to near-zero at the next grid point. It repeats. It's predictable. And if you know the quantization parameters, you can calculate it exactly.

This is fundamentally different from random noise. Random noise is hard to exploit because you can't predict it. Structured error is a pattern, and patterns can be learned. Keep this in mind — we'll come back to it in Part 3.

---

## Beyond Linear Quantization

Everything above uses uniform linear quantization — evenly spaced grid points, same Δ across all values. It's clean, it's simple, and it works. But it has a problem.

Real model weights don't distribute evenly. Large language models in particular tend to have most weights clustered near zero, with a few outliers far from the center. A uniform grid wastes most of its levels on the tails where few values live, and gives too-coarse resolution to the dense center where most values cluster.

A simple way to see this: try quantizing values that span several orders of magnitude. If your values range from 0.001 to 1.0, a 4-bit uniform grid has a step size of about 0.067 — larger than the smallest values themselves. Everything below 0.03 gets crushed to the same bucket. But if you quantize in log-space instead, the grid points spread geometrically — dense near the small values, sparse near the top. Small values get the relative precision they need.

```python
import math

BITS = 4
LEVELS = (1 << BITS) - 1  # 2^4 - 1 = 15 quantization levels

def uniform_quantize(v, lo, hi):
    """Quantize v into one of LEVELS evenly-spaced buckets between lo and hi."""
    step = (hi - lo) / LEVELS
    return lo + round((max(lo, min(hi, v)) - lo) / step) * step

def log_quantize(v, lo, hi):
    """Quantize v into one of LEVELS geometrically-spaced buckets.
    Maps v → log(v), quantizes uniformly in log-space, then maps back with exp."""
    log_lo, log_hi = math.log(lo), math.log(hi)
    step = (log_hi - log_lo) / LEVELS
    bucket = round((max(log_lo, min(log_hi, math.log(v))) - log_lo) / step)
    return math.exp(log_lo + bucket * step)

values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
lo, hi = values[0], values[-1]

print(f"{'Value':>6}  {'Uniform err':>11}  {'Log err':>8}")
print("-" * 32)
for v in values:
    ue = abs(v - uniform_quantize(v, lo, hi))
    le = abs(v - log_quantize(v, lo, hi))
    print(f"{v:>6.3f}  {ue:>11.6f}  {le:>8.6f}")
```

```
 Value  Uniform err   Log err
--------------------------------
 0.001     0.000000  0.000000
 0.005     0.004000  0.001019
 0.010     0.009000  0.000000
 0.050     0.017600  0.010189
 0.100     0.032400  0.000000
 0.250     0.017400  0.001189
 0.500     0.032800  0.101893
 0.750     0.016400  0.119043
 1.000     0.000000  0.000000
```

Look at the small values. At 0.005, uniform quantization has nearly 4× the error of log quantization. At 0.01, uniform is off by 0.009 — almost the value itself — while log hits it exactly. The tradeoff is that log quantization gives up precision at the top of the range, where values like 0.5 and 0.75 get coarser treatment. That's fine if your signal lives in the small values. Not fine if it doesn't. The right quantization scheme depends on where your values actually are.

This is where modern methods get clever.

**LLM.int8()** noticed that transformer weights have a small number of extreme outliers — values much larger than the rest. These outliers are rare but critical. Quantizing them with the same grid as everything else destroys them. The solution: split the matrix multiplication into two parts. The outlier channels stay in 16-bit. Everything else gets quantized to 8-bit. Two precisions, one operation, minimal loss. This was one of the first methods to make 8-bit inference practical for large transformers without noticeable quality degradation.

**SmoothQuant** approached the problem from the activation side. Weights are relatively easy to quantize — they're fixed after training and you can calibrate for them. Activations are harder because they change with every input. SmoothQuant found that you can shift the quantization difficulty from activations to weights by applying a per-channel scaling factor. It "smooths" the activation distribution at the cost of making weights slightly harder to quantize — but since weights are easier to handle in the first place, this is a good trade. The result: both weights and activations can be quantized to 8-bit with minimal impact.

**AWQ** (Activation-Aware Weight Quantization) took yet another angle. Instead of treating all weights equally, it observes which weight channels — columns of the weight matrix, each corresponding to one input feature — produce the largest activations — those are the ones that matter most. Protecting just 1% of channels (by keeping them at higher effective precision through scaling) can dramatically reduce quantization error. It's the quantization version of the Pareto principle: a small fraction of the weights do most of the work.

**QLoRA** combined quantization with fine-tuning. Take a pre-trained model, quantize it to 4 bits using a specialized format (NormalFloat4), then attach small trainable adapters (LoRA) in full precision. The base model is frozen and tiny. The adapters learn the task-specific adjustments. This made it possible to fine-tune a 65B parameter model on a single 48GB GPU — something that would normally require a cluster. Quantization here isn't just about inference. It's about making training accessible.

These methods share a philosophy: don't treat quantization as blind compression. Understand the structure of what you're compressing, and use that understanding to protect what matters.

---

## What's Coming Next

Now you know how quantization works — not just conceptually, but mechanically. You've seen what Δ does, where the information dies, and why the error has structure.

In **Part 3**, we'll zoom out. We'll look at how different model architectures respond to quantization — why transformers are sensitive but vision models are more robust. We'll explore what tiny, quantized models actually *enable*: edge AI, precision orchestration, portable agents. And we'll get into something that's rarely covered: the security implications of that structured error we just visualized.

Because a predictable error pattern isn't just a technical curiosity. It's an attack surface.

But that's for next time.

---

*This is Part 2 of a three-part series on quantization. [Part 1](part1.md) covers the fundamentals: what quantization is, what dequantization reveals, and why the error matters. [Part 3](part3.md) explores deployment strategies, edge AI, and the security surface of compressed models.*

---

## References

*This list is not exhaustive — there are many other sources with ongoing research and examples. Quantization is a fast-moving field, and we're always learning.*

[1] Liu, Z., et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." *arXiv preprint arXiv:2402.02750*, 2024. [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)

[2] DeepLearning.AI. "Quantization In Depth." Short course. [https://www.deeplearning.ai/short-courses/quantization-in-depth/](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

[3] Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314*, 2023. [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

[4] Lee, J., et al. "A Comprehensive Evaluation of Quantized Instruction-Tuned Large Language Models: An Experimental Analysis up to 405B." *arXiv preprint arXiv:2409.11055*, 2024. [https://arxiv.org/abs/2409.11055](https://arxiv.org/abs/2409.11055)

[5] vLLM Project. "Quantization." vLLM Documentation. [https://docs.vllm.ai/en/latest/features/quantization/](https://docs.vllm.ai/en/latest/features/quantization/)
