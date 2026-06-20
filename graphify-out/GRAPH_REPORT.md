# Graph Report - .  (2026-06-19)

## Corpus Check
- Corpus is ~11,120 words - fits in a single context window. You may not need a graph.

## Summary
- 131 nodes · 157 edges · 20 communities detected
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Attention Mechanism|Attention Mechanism]]
- [[_COMMUNITY_Shared Foundation|Shared Foundation]]
- [[_COMMUNITY_GPT Generation|GPT Generation]]
- [[_COMMUNITY_Masked LM Generation|Masked LM Generation]]
- [[_COMMUNITY_Testing & Validation|Testing & Validation]]
- [[_COMMUNITY_Positional Encoding|Positional Encoding]]
- [[_COMMUNITY_Tokenization|Tokenization]]
- [[_COMMUNITY_Documentation|Documentation]]
- [[_COMMUNITY_File Structure|File Structure]]
- [[_COMMUNITY_Embeddings|Embeddings]]
- [[_COMMUNITY_Concepts & Principles|Concepts & Principles]]
- [[_COMMUNITY_Design Decisions|Design Decisions]]
- [[_COMMUNITY_Component Connectivity|Component Connectivity]]
- [[_COMMUNITY_Graph Analysis|Graph Analysis]]
- [[_COMMUNITY_Performance Metrics|Performance Metrics]]
- [[_COMMUNITY_Integration|Integration]]
- [[_COMMUNITY_Paradigm Comparison|Paradigm Comparison]]
- [[_COMMUNITY_Implementation Details|Implementation Details]]
- [[_COMMUNITY_Reference Implementation|Reference Implementation]]
- [[_COMMUNITY_Thesis Validation|Thesis Validation]]

## God Nodes (most connected - your core abstractions)
1. `MultiHeadSelfAttention` - 18 edges
2. `TinyGPT` - 11 edges
3. `TinyMaskedLM` - 11 edges
4. `MultiHeadSelfAttention` - 10 edges
5. `scaled_dot_product_attention()` - 8 edges
6. `TransformerBlock` - 8 edges
7. `TransformerBlock` - 6 edges
8. `TransformerBlock` - 6 edges
9. `softmax()` - 6 edges
10. `TinyGPT` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Shared Multi-Head Attention Thesis` --demonstrates--> `Masked LM Paradigm (Bidirectional with Iterative Unmasking)`  [EXTRACTED]
  ARCHITECTURE.md → COMPLETION_SUMMARY.md
- `Autoregressive GPT: causal blocks + left-to-right sampling.  TinyGPT implements` --uses--> `MultiHeadSelfAttention`  [INFERRED]
  /Users/ashraf-macbookair/repos/projects/build9-ai-projects/transformers-vs-diffusion/src/transformer/gpt.py → /Users/ashraf-macbookair/repos/projects/build9-ai-projects/transformers-vs-diffusion/src/common/attention.py
- `Diffusion/masked LM: bidirectional blocks + iterative unmasking.  TinyMaskedLM i` --uses--> `MultiHeadSelfAttention`  [INFERRED]
  /Users/ashraf-macbookair/repos/projects/build9-ai-projects/transformers-vs-diffusion/src/diffusion/masked_lm.py → /Users/ashraf-macbookair/repos/projects/build9-ai-projects/transformers-vs-diffusion/src/common/attention.py
- `Shared Multi-Head Attention Thesis` --demonstrates--> `GPT Paradigm (Autoregressive with Causal Attention)`  [EXTRACTED]
  ARCHITECTURE.md → COMPLETION_SUMMARY.md
- `Execution Phase 6: Architecture Verification` --validates--> `Shared Multi-Head Attention Thesis`  [EXTRACTED]
  EXECUTION_LOG.md → ARCHITECTURE.md

## Hyperedges (group relationships)
- **Shared Module Foundation Hyperedge** — multihead_self_attention, token_positional_embedding, transformer_block, tinygpt, tiny_masked_lm [EXTRACTED 1.00]
- **GPT Paradigm Hyperedge** — tinygpt, causal_masking, autoregressive_generation, next_token_prediction, src_transformer_gpt [EXTRACTED 1.00]
- **Masked LM Paradigm Hyperedge** — tiny_masked_lm, bidirectional_attention, iterative_unmasking, masked_denoising, src_diffusion_masked_lm [EXTRACTED 1.00]

## Communities

### Community 0 - "Attention Mechanism"
Cohesion: 0.08
Nodes (14): multi_head_self_attention(), Pure-NumPy reference implementation of scaled dot-product & multi-head self-atte, q,k,v: (seq, d). Returns (out, weights) where out is (seq, d)., x: (seq, d_model). W*: (d_model, d_model). Returns (seq, d_model)., scaled_dot_product_attention(), softmax(), 20-test suite for the Transformers vs Diffusion study project.  The char tokeniz, test_attention_output_shape_matches_input() (+6 more)

### Community 1 - "Shared Foundation"
Cohesion: 0.11
Nodes (16): MultiHeadSelfAttention, Multi-head self-attention, shared by GPT and the masked LM.  This module is the, Multi-head self-attention with optional causal masking.      Implements scaled d, x: (batch, seq, d_model) -> (batch, seq, d_model), Generate autoregressive sequence.          Args:             idx: (batch, seq) s, Single transformer block: causal attention + feedforward with residuals.      Us, x: (batch, seq, d_model) -> (batch, seq, d_model), Tiny autoregressive GPT: embeddings + causal transformer blocks + language head. (+8 more)

### Community 2 - "GPT Generation"
Cohesion: 0.12
Nodes (22): Shared Multi-Head Attention Thesis, Autoregressive Generation, Causal Masking, Configuration Matters (Causal Flag as Critical Pivot), Execution Phase 6: Architecture Verification, FeedForward Block, God Nodes: Highest Connectivity Components, GPT Convergence: Rapid (next-token prediction is simpler) (+14 more)

### Community 3 - "Masked LM Generation"
Cohesion: 0.15
Nodes (10): generate(), Autoregressive GPT: causal blocks + left-to-right sampling.  TinyGPT implements, token_ids: (batch, seq) -> logits: (batch, seq, vocab_size), generate(), Diffusion/masked LM: bidirectional blocks + iterative unmasking.  TinyMaskedLM i, token_ids: (batch, seq) -> logits: (batch, seq, vocab_size), Sinusoidal Positional Encoding, src/common/attention.py (+2 more)

### Community 4 - "Testing & Validation"
Cohesion: 0.2
Nodes (10): Bidirectional Attention, CharTokenizer, Iterative Unmasking (Diffusion-style), MASK_TOKEN_ID = 0 Convention, Masked-Token Denoising (Masked LM Training Objective), Masked LM Convergence: Slower (denoising with random masking), Masked LM Paradigm (Bidirectional with Iterative Unmasking), src/common/tokenizer.py (+2 more)

### Community 5 - "Positional Encoding"
Cohesion: 0.25
Nodes (6): build_embeddings(), Token + positional embeddings for both GPT and masked-LM.  This module provides, Combines token embeddings with sinusoidal positional encoding.      Learns a tok, token_ids: (batch, seq) -> (batch, seq, d_model), Factory function returning an embedding layer.      Args:         vocab_size: nu, TokenPositionalEmbedding

### Community 6 - "Tokenization"
Cohesion: 0.67
Nodes (3): Attention Reference Tests (7 passing), Test Suite: 15 Passing Tests, Tokenizer Tests (8 passing)

### Community 11 - "Documentation"
Cohesion: 1.0
Nodes (1): Knowledge Graph: 106 Nodes

### Community 12 - "File Structure"
Cohesion: 1.0
Nodes (1): Knowledge Graph: 134 Edges

### Community 13 - "Embeddings"
Cohesion: 1.0
Nodes (1): Knowledge Graph: 3 Hyperedges

### Community 14 - "Concepts & Principles"
Cohesion: 1.0
Nodes (1): Execution Phase 1: Setup

### Community 15 - "Design Decisions"
Cohesion: 1.0
Nodes (1): Execution Phase 2: Forward Pass (Untrained)

### Community 16 - "Component Connectivity"
Cohesion: 1.0
Nodes (1): Execution Phase 3: Training (50 iterations)

### Community 17 - "Graph Analysis"
Cohesion: 1.0
Nodes (1): Execution Phase 4: Generation - GPT (Autoregressive)

### Community 18 - "Performance Metrics"
Cohesion: 1.0
Nodes (1): Execution Phase 5: Generation - Masked LM (Iterative Unmasking)

### Community 19 - "Integration"
Cohesion: 1.0
Nodes (1): notebooks/01_attention.ipynb

### Community 20 - "Paradigm Comparison"
Cohesion: 1.0
Nodes (1): notebooks/02_gpt.ipynb

### Community 21 - "Implementation Details"
Cohesion: 1.0
Nodes (1): notebooks/03_diffusion_lm.ipynb

### Community 22 - "Reference Implementation"
Cohesion: 1.0
Nodes (1): notebooks/04_comparison.ipynb

### Community 23 - "Thesis Validation"
Cohesion: 1.0
Nodes (1): tests/test_suite.py

## Knowledge Gaps
- **50 isolated node(s):** `20-test suite for the Transformers vs Diffusion study project.  The char tokeniz`, `x: (batch, seq, d_model) -> (batch, seq, d_model)`, `token_ids: (batch, seq) -> logits: (batch, seq, vocab_size)`, `Generate autoregressive sequence.          Args:             idx: (batch, seq) s`, `x: (batch, seq, d_model) -> (batch, seq, d_model)` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Documentation`** (1 nodes): `Knowledge Graph: 106 Nodes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `File Structure`** (1 nodes): `Knowledge Graph: 134 Edges`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Embeddings`** (1 nodes): `Knowledge Graph: 3 Hyperedges`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Concepts & Principles`** (1 nodes): `Execution Phase 1: Setup`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Design Decisions`** (1 nodes): `Execution Phase 2: Forward Pass (Untrained)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Component Connectivity`** (1 nodes): `Execution Phase 3: Training (50 iterations)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Graph Analysis`** (1 nodes): `Execution Phase 4: Generation - GPT (Autoregressive)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Performance Metrics`** (1 nodes): `Execution Phase 5: Generation - Masked LM (Iterative Unmasking)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Integration`** (1 nodes): `notebooks/01_attention.ipynb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Paradigm Comparison`** (1 nodes): `notebooks/02_gpt.ipynb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Implementation Details`** (1 nodes): `notebooks/03_diffusion_lm.ipynb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reference Implementation`** (1 nodes): `notebooks/04_comparison.ipynb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Thesis Validation`** (1 nodes): `tests/test_suite.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `softmax()` connect `Attention Mechanism` to `Shared Foundation`, `Masked LM Generation`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `MultiHeadSelfAttention` connect `Shared Foundation` to `Masked LM Generation`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `MultiHeadSelfAttention` connect `GPT Generation` to `Masked LM Generation`, `Testing & Validation`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `MultiHeadSelfAttention` (e.g. with `TransformerBlock` and `TinyGPT`) actually correct?**
  _`MultiHeadSelfAttention` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `scaled_dot_product_attention()` (e.g. with `test_attention_output_shape_matches_input()` and `test_attention_weights_are_row_stochastic()`) actually correct?**
  _`scaled_dot_product_attention()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `20-test suite for the Transformers vs Diffusion study project.  The char tokeniz`, `x: (batch, seq, d_model) -> (batch, seq, d_model)`, `token_ids: (batch, seq) -> logits: (batch, seq, vocab_size)` to the rest of the system?**
  _50 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Attention Mechanism` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._