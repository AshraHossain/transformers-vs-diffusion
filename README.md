# Project 8 — Transformers & Diffusion LLMs: What's the Connection?

A from-scratch study project demonstrating that two fundamentally different generation paradigms—autoregressive (GPT-style) and diffusion (masked-LM-style)—rest on **identical transformer foundations**. The thesis: configuration and training objective matter; architecture doesn't.

## Stack

- **Framework:** PyTorch (implemented from scratch — no HF model classes)
- **Workflow:** Jupyter notebooks for narrative + `src/` for reusable modules
- **Scope:** char-level / tiny-vocab toy models that train on CPU or a small GPU
- **No external model APIs** — this is about understanding internals

## Core Thesis: Same Foundation, Different Objectives

| Aspect | Autoregressive (GPT) | Diffusion / Masked (LLaDA-style) |
|--------|----------------------|----------------------------------|
| **Training Objective** | Next-token prediction | Masked-token denoising |
| **Attention Mode** | Causal (left-to-right only) | Bidirectional (full context) |
| **Generation Strategy** | Sequential left-to-right | Iterative unmasking (diffusion-style) |
| **Shared Foundation** | `MultiHeadSelfAttention` + `TokenPositionalEmbedding` | Same exact classes |
| **Single Difference** | `causal=True` flag | `causal=False` flag |

**The insight:** A single boolean flag in the attention layer enables two completely different paradigms. Both models use:
- Identical `MultiHeadSelfAttention` (scaled dot-product attention with optional causal masking)
- Identical `TokenPositionalEmbedding` (sinusoidal positional encoding)
- Identical `TransformerBlock` (pre-norm LayerNorm → Attention → Residual → FF → Residual)

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run notebooks in order
jupyter lab
# 01_attention.ipynb     → Attention mechanism fundamentals
# 02_gpt.ipynb           → Train & generate with autoregressive GPT
# 03_diffusion_lm.ipynb  → Train & generate with masked-LM (diffusion-style)
# 04_comparison.ipynb    → Side-by-side comparison, thesis verification
```

## Project Structure

```
src/
  common/
    attention.py         # MultiHeadSelfAttention (the shared core, 18 edge connections)
    embeddings.py        # TokenPositionalEmbedding (identical for both models)
    tokenizer.py         # CharTokenizer (char-level vocabulary)
    sdpa_reference.py    # NumPy reference implementation (for validation)
  
  transformer/
    gpt.py              # TinyGPT: causal attention + autoregressive sampling
  
  diffusion/
    masked_lm.py        # TinyMaskedLM: bidirectional attention + iterative unmasking

notebooks/
  01_attention.ipynb     # Scaled dot-product attention mechanics
  02_gpt.ipynb           # Autoregressive language modeling
  03_diffusion_lm.ipynb  # Iterative denoising generation
  04_comparison.ipynb    # Direct comparison, shared module verification

tests/
  test_suite.py         # 15 passing tests covering all modules
```

## Key Components by Connectivity (God Nodes)

From the knowledge graph analysis:

1. **`MultiHeadSelfAttention`** (18 connections) — The lynchpin shared by both models
   - Used in both GPT (causal=True) and Masked LM (causal=False)
   - Implements scaled dot-product attention: Attention(Q,K,V) = softmax((Q@K^T)/√d_k) @ V
   - Single `causal` flag enables bidirectional ↔ causal switching

2. **`TinyGPT`** (10 connections) — Autoregressive paradigm
   - Stacks causal TransformerBlocks
   - Generates left-to-right via sequential token prediction

3. **`TinyMaskedLM`** (10 connections) — Diffusion paradigm
   - Stacks bidirectional TransformerBlocks
   - Generates via iterative unmasking with confidence-based refinement

4. **`scaled_dot_product_attention()`** (8 connections) — Core mechanism
   - NumPy reference implementation for validation
   - PyTorch implementation in MultiHeadSelfAttention

## Why This Project Matters

This is **pure research/education**. The thesis is concrete: you can implement two completely different generation paradigms by:
1. Using identical transformer foundations
2. Toggling a single configuration flag (causal masking)
3. Changing only the training objective and sampling strategy

No hand-waving about "the same underlying architecture." The code proves it—the shared `src/common/attention.py` is imported identically by both models.
