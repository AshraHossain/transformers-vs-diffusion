# Transformers vs Diffusion: Shared Architecture

## The Thesis

Both autoregressive (GPT) and diffusion (masked-LM) generation paradigms rest on **the same multi-head self-attention block**. The only difference is configuration:

| Aspect | GPT | Masked LM |
|--------|-----|-----------|
| **Attention** | `MultiHeadSelfAttention(causal=True)` | `MultiHeadSelfAttention(causal=False)` |
| **Generation** | Autoregressive (predict next token) | Iterative unmasking (diffusion-style) |
| **Training** | Next-token prediction | Masked token prediction (denoising) |

This repository demonstrates this thesis with minimal, from-scratch implementations in PyTorch.

---

## Shared Modules

### `src/common/attention.py` — MultiHeadSelfAttention

The shared foundation. Implements scaled dot-product attention with optional causal masking.

**Key equations:**
```
Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
```

With causal mask (GPT):
```
scores[i, j] = -inf  if j > i  (blocks future positions)
```

**Usage:**
```python
from src.common.attention import MultiHeadSelfAttention

# GPT: causal attention
attn_gpt = MultiHeadSelfAttention(d_model=64, n_heads=4, causal=True)
out = attn_gpt(embeddings)  # (batch, seq, d_model)

# Masked LM: bidirectional attention
attn_masked = MultiHeadSelfAttention(d_model=64, n_heads=4, causal=False)
out = attn_masked(embeddings)  # (batch, seq, d_model)
```

**Reference implementation:** `src/common/sdpa_reference.py`
- Pure NumPy version for verification
- Used in test suite to validate PyTorch implementation

---

### `src/common/embeddings.py` — TokenPositionalEmbedding

Shared embedding layer combining:
1. **Token embeddings** — learned lookup table (vocab_size × d_model)
2. **Positional encoding** — sinusoidal functions (non-trainable)

The sinusoidal encoding allows sequences longer than training examples:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Usage:**
```python
from src.common.embeddings import build_embeddings

embed = build_embeddings(vocab_size=256, d_model=64, max_len=512)
token_ids = torch.tensor([[5, 3, 7, 2]])  # (batch=1, seq=4)
embeddings = embed(token_ids)  # (batch=1, seq=4, d_model=64)
```

---

### `src/common/tokenizer.py` — CharTokenizer

Character-level tokenizer for toy corpora. Used in notebooks and tests.

```python
from src.common.tokenizer import CharTokenizer

tokenizer = CharTokenizer("hello world")
tokens = tokenizer.encode("hello")  # [3, 2, 4, 4, 5]
text = tokenizer.decode(tokens)     # "hello"
```

---

## Model Architecture

Both models follow the same pattern:

```
Token IDs (batch, seq)
    ↓
[Embeddings] ← shared via build_embeddings()
    ↓
[TransformerBlock] × n_layers
    ├─ LayerNorm
    ├─ MultiHeadSelfAttention ← shared, with causal flag
    ├─ Residual
    ├─ LayerNorm
    ├─ FeedForward (2 linear layers)
    └─ Residual
    ↓
[LayerNorm]
    ↓
[Head] (d_model → vocab_size)
    ↓
Logits (batch, seq, vocab_size)
```

### Differences

**GPT** (`src/transformer/gpt.py`):
- Uses `causal=True` attention
- Predicts next token: `loss = CE(logits[:, :-1], targets[:, 1:])`
- Generates left-to-right: sample token, append, repeat

**Masked LM** (`src/diffusion/masked_lm.py`):
- Uses `causal=False` attention
- Predicts masked tokens: `loss = CE(logits[mask_pos], original[mask_pos])`
- Generates iteratively: unmask confident predictions, repeat

---

## How to Use Both Models

### Train GPT

```python
import torch
import torch.nn.functional as F
from src.transformer.gpt import TinyGPT
from src.common.tokenizer import CharTokenizer

# Setup
tokenizer = CharTokenizer("hello world")
gpt = TinyGPT(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2)
optimizer = torch.optim.Adam(gpt.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    token_ids = torch.tensor([tokenizer.encode("hello world")])
    
    logits = gpt(token_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, gpt.vocab_size),
        token_ids[:, 1:].reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generation
start = torch.tensor([[tokenizer.stoi['h']]])
generated = gpt.generate(start, max_new_tokens=10)
text = tokenizer.decode(generated[0].tolist())
print(f"Generated: {text}")
```

### Train Masked LM

```python
import torch
import torch.nn.functional as F
from src.diffusion.masked_lm import TinyMaskedLM
from src.common.tokenizer import CharTokenizer

# Setup
tokenizer = CharTokenizer("hello world")
# vocab_size + 1 because ID 0 is reserved for MASK
masked_lm = TinyMaskedLM(
    vocab_size=tokenizer.vocab_size + 1,
    d_model=64, n_layers=2
)
optimizer = torch.optim.Adam(masked_lm.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    tokens_raw = tokenizer.encode("hello world")
    tokens = torch.tensor([[t + 1 for t in tokens_raw]])  # Shift (0 = MASK)
    
    # Mask 50% of tokens
    masked_input = tokens.clone()
    mask = torch.rand_like(tokens, dtype=torch.float) < 0.5
    masked_input[mask] = 0
    
    logits = masked_lm(masked_input)
    loss = F.cross_entropy(
        logits[mask].reshape(-1, masked_lm.vocab_size),
        tokens[mask].reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generation
generated = masked_lm.generate(length=11, steps=4)
# Shift back
tokens = [t.item() - 1 for t in generated[0] if t > 0]
text = tokenizer.decode(tokens)
print(f"Generated: {text}")
```

---

## Directory Structure

```
src/
├── common/                    # Shared by both models
│   ├── __init__.py
│   ├── attention.py          # MultiHeadSelfAttention (causal flag)
│   ├── embeddings.py         # TokenPositionalEmbedding
│   ├── tokenizer.py          # CharTokenizer
│   └── sdpa_reference.py     # NumPy reference (for testing)
├── transformer/              # Autoregressive GPT
│   ├── __init__.py
│   └── gpt.py               # TinyGPT (causal=True)
└── diffusion/               # Masked LM / Diffusion
    ├── __init__.py
    └── masked_lm.py         # TinyMaskedLM (causal=False)

notebooks/
├── 01_attention.ipynb       # Understand scaled dot-product attention
├── 02_gpt.ipynb            # Train and generate with GPT
├── 03_diffusion_lm.ipynb   # Train and generate with masked LM
└── 04_comparison.ipynb     # Side-by-side comparison

tests/
└── test_suite.py           # 20 tests validating core components
```

---

## Key Insights

1. **Shared foundation** — The `MultiHeadSelfAttention` module is identical; only the `causal` flag differs.

2. **Dual objectives** — GPT learns next-token prediction; masked-LM learns denoising.

3. **Dual generation** — GPT generates left-to-right; masked-LM generates by progressively unmasking.

4. **Same building blocks** — Both use the same embeddings, layer norm, feedforward, and residual connections.

This demonstrates that the core transformer machinery is paradigm-agnostic: it's the training objective and sampling strategy that define the generation approach.

---

## Running Tests

```bash
# Run all tests
pytest tests/test_suite.py -v

# Run specific test
pytest tests/test_suite.py::test_causal_mask_blocks_future_positions -v
```

**Expected output:** 15 passed (reference tests), 7 skipped (model tests until notebooks are run).

---

## References

- **Transformer paper** — "Attention Is All You Need" (Vaswani et al., 2017)
- **GPT paper** — "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **Masked LM** — "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- **Diffusion in NLP** — "Diffusion-LM Improves Controllable Text Generation" (Li et al., 2022)
