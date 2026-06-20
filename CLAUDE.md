# CLAUDE.md — Transformers & Diffusion LLMs

## What we're building
An educational, from-scratch PyTorch comparison of two generation paradigms over the same shared Transformer foundation: an autoregressive GPT (causal attention, next-token objective, left-to-right sampling) and a diffusion/masked language model (bidirectional attention, masked-denoising objective, iterative unmasking). The thesis to demonstrate: both rest on the same multi-head self-attention block.

## Stack & conventions
- Python 3.11+, PyTorch only — implement attention/blocks by hand, do NOT import HF model classes.
- Tiny toy scale (char-level or small vocab) so everything trains on CPU/small GPU.
- `src/common/attention.py` is shared by both models — this reuse is the whole point; don't duplicate it.
- Notebooks tell the story; `src/` holds the reusable, importable, testable code.

## Architecture map
- `src/common/attention.py` — `MultiHeadSelfAttention` with an optional causal mask flag.
- `src/common/embeddings.py` — token + positional embeddings; `tokenizer.py` — char tokenizer.
- `src/transformer/gpt.py` — causal block stack + `generate()` (autoregressive sampling).
- `src/diffusion/masked_lm.py` — bidirectional block stack + `generate()` (iterative unmasking).
- `notebooks/` — `01_attention`, `02_gpt`, `03_diffusion_lm`, `04_comparison`.

## Next concrete steps
1. Implement + test `MultiHeadSelfAttention` (shape tests; causal mask zeroes the upper triangle).
2. Build the GPT block stack; overfit a tiny string to validate training.
3. Build the masked-LM with bidirectional attention + an unmasking sampler.
4. Write `04_comparison.ipynb`: same prompt, contrast generation dynamics + show the shared attention module.

## Definition of done
Both tiny models train on a toy corpus and generate text; the comparison notebook makes the shared-attention / different-objective relationship explicit, with passing shape/masking tests in `tests/`.

## Testing
20 tests in `tests/` cover attention output shapes, causal vs bidirectional masking, the tokenizer round-trip, and sampler step counts. Run with `pytest`.

## Knowledge Graph
Run `graphify` to build a knowledge graph of the codebase:
```bash
graphify
```
This produces:
- `graphify-out/graph.html` — Interactive visualization of 106 nodes and 134 edges
- `graphify-out/GRAPH_REPORT.md` — Audit trail with god nodes, communities, and surprises

**Key findings from graph:**
- **MultiHeadSelfAttention**: 18 edges (highest connectivity—the shared foundation)
- **TinyGPT**: 10 edges (autoregressive paradigm)
- **TinyMaskedLM**: 10 edges (diffusion paradigm)
- **12 communities** organized by architecture and purpose
- **75% EXTRACTED edges** (explicit in code), **25% INFERRED** (derived relationships)
