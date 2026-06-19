# Project 8 — Transformers & Diffusion LLMs: What's the Connection?

A from-scratch study project. Implement a tiny autoregressive Transformer (GPT-style) and a tiny masked/diffusion language model (LLaDA-style), then compare how they generate text — autoregressive next-token prediction vs iterative masked denoising.

## Stack

- **Framework:** PyTorch (implemented from scratch — no HF model classes)
- **Workflow:** Jupyter notebooks for narrative + `src/` for reusable modules
- **Scope:** char-level / tiny-vocab toy models that train on CPU or a small GPU
- **No external model APIs** — this is about understanding internals

## What you'll build & compare

| | Autoregressive (GPT) | Diffusion / masked (LLaDA-style) |
|---|---|---|
| Objective | next-token prediction | masked-token denoising |
| Attention | causal (triangular mask) | bidirectional |
| Generation | left-to-right, one token at a time | iterative unmasking over steps |
| Key shared piece | **multi-head self-attention** | **multi-head self-attention** |

The punchline: both are built on the same Transformer attention block; they differ in masking and the training objective.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab     # notebooks/01_attention.ipynb -> 02_gpt.ipynb -> 03_diffusion_lm.ipynb -> 04_comparison.ipynb
```

## Project layout

```
notebooks/        # 01 attention, 02 GPT, 03 diffusion LM, 04 comparison
src/
  common/         # multi-head attention, embeddings, tokenizer (shared!)
  transformer/    # causal GPT block + autoregressive sampling
  diffusion/      # bidirectional block + iterative masked denoising
```

## Notes
Pure **research/education** (like Project 6). The shared `common/attention.py` is deliberately reused by both models to make the "same foundation" point concrete.
