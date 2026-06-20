# Usage Guide: Shared Modules in GPT & Masked LM

This guide shows how to import and use the shared modules in both models. The key insight: both models use **identical components**—only the `causal` flag differs in attention, and the training objective differs.

**From the knowledge graph:** The shared modules (MultiHeadSelfAttention: 18 edges, TokenPositionalEmbedding: 6 edges) are the highest-connectivity nodes in the codebase, proving they're the architectural foundation.

---

## Shared Module Imports

### MultiHeadSelfAttention

Used by both GPT and Masked LM, configured via the `causal` flag:

```python
from src.common.attention import MultiHeadSelfAttention
import torch

# For GPT: causal attention (left-to-right only)
attn_gpt = MultiHeadSelfAttention(
    d_model=64,
    n_heads=4,
    causal=True  # Blocks future positions
)

# For Masked LM: bidirectional attention
attn_masked = MultiHeadSelfAttention(
    d_model=64,
    n_heads=4,
    causal=False  # Allows attention in both directions
)

# Both have identical forward signature
x = torch.randn(2, 10, 64)  # (batch=2, seq=10, d_model=64)
out_gpt = attn_gpt(x)       # (batch=2, seq=10, d_model=64)
out_masked = attn_masked(x) # (batch=2, seq=10, d_model=64)
```

### TokenPositionalEmbedding

Shared embedding layer used by both models:

```python
from src.common.embeddings import build_embeddings
import torch

# Create embedding layer (identical for both models)
embed = build_embeddings(
    vocab_size=256,
    d_model=64,
    max_len=512
)

# Both models use it the same way
token_ids = torch.tensor([[5, 3, 7, 2]])
embeddings = embed(token_ids)  # (1, 4, 64)
```

### CharTokenizer

Character-level tokenizer for toy corpora:

```python
from src.common.tokenizer import CharTokenizer

tokenizer = CharTokenizer("hello world")
print(f"Vocab size: {tokenizer.vocab_size}")        # 8
print(f"Vocab: {tokenizer.chars}")                  # ['', ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']

# Encode/decode
tokens = tokenizer.encode("hello")
print(tokens)                                       # [4, 5, 6, 6, 7]
text = tokenizer.decode(tokens)
print(text)                                         # "hello"
```

---

## Model Setup & Usage

### TinyGPT

Autoregressive generation with causal attention:

```python
from src.transformer.gpt import TinyGPT
from src.common.tokenizer import CharTokenizer
import torch
import torch.nn.functional as F

# Initialize tokenizer and model
tokenizer = CharTokenizer("the quick brown fox")
gpt = TinyGPT(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_layers=2,
    n_heads=4,
    max_len=512
)

# Forward pass: get logits
token_ids = torch.tensor([tokenizer.encode("the")])
logits = gpt(token_ids)
print(logits.shape)  # (1, 3, vocab_size)

# Generate: autoregressive sampling
generated = gpt.generate(
    idx=token_ids,
    max_new_tokens=10,
    temperature=1.0  # Stochastic sampling
)
text = tokenizer.decode(generated[0].tolist())
print(f"Generated: {text}")

# Training
optimizer = torch.optim.Adam(gpt.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    logits = gpt(token_ids)
    # Loss: predict next token at each position
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, gpt.vocab_size),
        token_ids[:, 1:].reshape(-1)
    )
    loss.backward()
    optimizer.step()
```

### TinyMaskedLM

Iterative generation with bidirectional attention:

```python
from src.diffusion.masked_lm import TinyMaskedLM
from src.common.tokenizer import CharTokenizer
import torch
import torch.nn.functional as F

# Initialize tokenizer and model
# Note: ID 0 is reserved for MASK token, so vocab_size is +1
tokenizer = CharTokenizer("the quick brown fox")
masked_lm = TinyMaskedLM(
    vocab_size=tokenizer.vocab_size + 1,  # +1 for MASK
    d_model=64,
    n_layers=2,
    n_heads=4,
    max_len=512
)

# Forward pass: get logits for masked positions
# We need to shift token IDs by 1 (since 0 = MASK)
token_ids = torch.tensor([tokenizer.encode("the")])
token_ids_shifted = token_ids + 1

# Create masked input (mask some positions)
masked_input = token_ids_shifted.clone()
masked_input[0, 1:] = 0  # Mask positions 1 and 2
logits = masked_lm(masked_input)
print(logits.shape)  # (1, 3, vocab_size)

# Generate: iterative unmasking
generated = masked_lm.generate(
    length=10,
    steps=4,        # Number of denoising iterations
    temperature=1.0
)
# Shift back from vocab_id to token_id
tokens = [t.item() - 1 for t in generated[0] if t > 0]
text = tokenizer.decode(tokens)
print(f"Generated: {text}")

# Training: masked denoising objective
optimizer = torch.optim.Adam(masked_lm.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    
    # Create random mask (50% masking rate)
    masked_input = token_ids_shifted.clone()
    mask = torch.rand_like(token_ids_shifted, dtype=torch.float) < 0.5
    masked_input[mask] = 0
    
    logits = masked_lm(masked_input)
    
    # Loss: predict original tokens at masked positions only
    loss = F.cross_entropy(
        logits[mask].reshape(-1, masked_lm.vocab_size),
        token_ids_shifted[mask].reshape(-1)
    )
    
    loss.backward()
    optimizer.step()
```

---

## Comparing Both Models

### Side-by-Side Training

```python
from src.transformer.gpt import TinyGPT
from src.diffusion.masked_lm import TinyMaskedLM
from src.common.tokenizer import CharTokenizer
import torch
import torch.nn.functional as F

tokenizer = CharTokenizer("hello world")

# Create both models with identical architecture
gpt = TinyGPT(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4)
masked_lm = TinyMaskedLM(vocab_size=tokenizer.vocab_size + 1, d_model=64, n_layers=2, n_heads=4)

# Token preparation
tokens = torch.tensor([tokenizer.encode("hello")])
tokens_shifted = tokens + 1

gpt_optim = torch.optim.Adam(gpt.parameters(), lr=0.01)
mlm_optim = torch.optim.Adam(masked_lm.parameters(), lr=0.01)

# Train for 50 steps
for step in range(50):
    # --- Train GPT ---
    gpt_optim.zero_grad()
    gpt_logits = gpt(tokens)
    gpt_loss = F.cross_entropy(
        gpt_logits[:, :-1].reshape(-1, gpt.vocab_size),
        tokens[:, 1:].reshape(-1)
    )
    gpt_loss.backward()
    gpt_optim.step()
    
    # --- Train Masked LM ---
    mlm_optim.zero_grad()
    masked_input = tokens_shifted.clone()
    mask = torch.rand_like(tokens_shifted, dtype=torch.float) < 0.5
    masked_input[mask] = 0
    mlm_logits = masked_lm(masked_input)
    mlm_loss = F.cross_entropy(
        mlm_logits[mask].reshape(-1, masked_lm.vocab_size),
        tokens_shifted[mask].reshape(-1)
    )
    mlm_loss.backward()
    mlm_optim.step()
    
    if step % 10 == 0:
        print(f"Step {step}: GPT loss={gpt_loss.item():.4f}, MLM loss={mlm_loss.item():.4f}")

# Generate from both
print("\nGeneration:")
start = torch.tensor([[tokenizer.stoi['h']]])
gpt_gen = gpt.generate(start, max_new_tokens=5)
mlm_gen = masked_lm.generate(length=6, steps=4)

print(f"GPT:       {tokenizer.decode(gpt_gen[0].tolist())}")
print(f"Masked LM: {tokenizer.decode([t.item()-1 for t in mlm_gen[0] if t > 0])}")
```

### Verify Shared Attention

```python
from src.transformer.gpt import TinyGPT
from src.diffusion.masked_lm import TinyMaskedLM

# Create models
gpt = TinyGPT(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
masked_lm = TinyMaskedLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4)

# Verify they use the same MultiHeadSelfAttention class
print(f"GPT block 0 attention type: {type(gpt.blocks[0].attn)}")
print(f"Masked LM block 0 attention type: {type(masked_lm.blocks[0].attn)}")
# Both print: <class 'src.common.attention.MultiHeadSelfAttention'>

# Verify causal flag differs
print(f"GPT attention causal: {gpt.blocks[0].attn.causal}")        # True
print(f"Masked LM attention causal: {masked_lm.blocks[0].attn.causal}")  # False

# Both use the same embeddings factory
print(f"GPT embeddings type: {type(gpt.embed)}")
print(f"Masked LM embeddings type: {type(masked_lm.embed)}")
# Both print: <class 'src.common.embeddings.TokenPositionalEmbedding'>
```

---

## Common Patterns

### Custom Hyperparameters

```python
from src.transformer.gpt import TinyGPT
from src.diffusion.masked_lm import TinyMaskedLM

# Larger model
gpt_large = TinyGPT(
    vocab_size=256,
    d_model=256,      # Larger hidden dimension
    n_layers=6,       # More layers
    n_heads=8,        # More heads (256/8 = 32 per head)
    max_len=2048      # Longer context
)

masked_lm_large = TinyMaskedLM(
    vocab_size=257,   # 256 + 1 for MASK
    d_model=256,
    n_layers=6,
    n_heads=8,
    max_len=2048
)

# Smaller model (for testing)
gpt_small = TinyGPT(vocab_size=50, d_model=32, n_layers=1, n_heads=2)
masked_lm_small = TinyMaskedLM(vocab_size=51, d_model=32, n_layers=1, n_heads=2)
```

### Batch Processing

```python
import torch
from src.transformer.gpt import TinyGPT

gpt = TinyGPT(vocab_size=100, d_model=64, n_layers=2, n_heads=4)

# Process multiple sequences in parallel
batch = torch.randint(0, 100, (4, 20))  # (batch=4, seq=20)
logits = gpt(batch)  # (batch=4, seq=20, vocab_size=100)

# Generate from batch of starting sequences
starts = torch.tensor([[5, 3], [7, 2], [9, 1], [4, 8]])
generated = gpt.generate(starts, max_new_tokens=10)
# generates 4 sequences in parallel
print(generated.shape)  # (4, 12)
```

### Saving & Loading

```python
import torch
from src.transformer.gpt import TinyGPT

gpt = TinyGPT(vocab_size=100, d_model=64, n_layers=2, n_heads=4)

# Save
torch.save(gpt.state_dict(), "gpt_model.pt")

# Load
gpt_loaded = TinyGPT(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
gpt_loaded.load_state_dict(torch.load("gpt_model.pt"))
```

---

## Testing

All core functionality is tested:

```bash
# Run full test suite
pytest tests/test_suite.py -v

# Test specific module
pytest tests/test_suite.py -k attention -v
pytest tests/test_suite.py -k tokenizer -v
```

**Test coverage:**
- ✓ Tokenizer round-trip (encode/decode)
- ✓ Attention output shapes
- ✓ Causal masking (upper triangle = 0)
- ✓ Bidirectional attention (not blocked)
- ✓ Multi-head configuration validation
- ✓ Positional encoding shapes

---

## Troubleshooting

**Q: Why use `causal=True` vs `causal=False`?**
- `causal=True` for autoregressive models (GPT): each position predicts the next
- `causal=False` for masked models (BERT/MLM): each position sees full context

**Q: Why shift tokens by 1 for masked LM?**
- ID 0 is reserved for the MASK token
- Tokens from the tokenizer are 0-indexed; we shift them to 1-indexed to avoid collision

**Q: How do I adjust generation quality?**
- **Temperature**: lower = greedier (argmax), higher = more random
- **Steps** (masked LM): more steps = more refinement iterations
- **max_new_tokens** (GPT): controls length

**Q: Can I use these models on GPU?**
- Yes! Both are standard PyTorch modules:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  gpt = gpt.to(device)
  ```
