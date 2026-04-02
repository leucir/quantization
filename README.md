# Quantization: Intelligence Survives Compression

A practical, code-first companion project for the 3-part quantization series.

Published post:
- [Intelligence Survives Compression — But Not Unchanged](https://leucir.substack.com/p/intelligence-survives-compression)

## What's in this repository

- `part1.md` - intuition, tradeoffs, and modern quantization methods
- `part2.md` - formula breakdown and implementation from scratch
- `part3.md` - deployment implications, architecture behavior, and security considerations
- `src/` - small runnable Python examples used in the series
- `diagrams/` - visual assets and Excalidraw sources

## Run the examples

This project uses Python and NumPy.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

Run the scripts:

```bash
python src/single.py
python src/error.py
python src/log.py
python src/2Bit.py
```

## Why this project exists

Quantization is often explained as a storage trick. This repo focuses on the full picture:
how precision compression changes model behavior, where it helps in practice, and what
tradeoffs appear when models move from cloud-heavy setups to edge deployment.
