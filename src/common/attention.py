"""Multi-head self-attention, shared by GPT and the masked LM.

This module is the cornerstone of the project's thesis: both autoregressive (GPT)
and diffusion (masked LM) generation paradigms rest on the same multi-head
self-attention block. The only difference is the `causal` flag:

  - GPT uses causal=True: blocks future positions, enforces left-to-right order
  - Masked LM uses causal=False: bidirectional attention, allows any position to attend

Scaled dot-product attention formula:
    Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V

With optional causal mask that sets scores to -inf for future positions.

Example usage:
    # For GPT (autoregressive)
    attn_gpt = MultiHeadSelfAttention(d_model=64, n_heads=4, causal=True)
    logits = attn_gpt(embeddings)  # (batch, seq, d_model)

    # For masked LM (bidirectional)
    attn_masked = MultiHeadSelfAttention(d_model=64, n_heads=4, causal=False)
    logits = attn_masked(embeddings)  # (batch, seq, d_model)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking.

    Implements scaled dot-product attention with learnable Q, K, V projections
    and optional causal masking for autoregressive models.

    Args:
        d_model: Embedding dimension. Must be divisible by n_heads.
        n_heads: Number of attention heads.
        causal: If True, apply causal mask to block future positions.
                GPT uses causal=True; masked LM uses causal=False.

    Raises:
        ValueError: If d_model is not divisible by n_heads.

    Shape:
        Input: (batch, seq, d_model)
        Output: (batch, seq, d_model)

    Example:
        >>> attn = MultiHeadSelfAttention(d_model=64, n_heads=4, causal=True)
        >>> x = torch.randn(2, 10, 64)  # (batch=2, seq=10, d_model=64)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 10, 64])
    """
    def __init__(self, d_model: int, n_heads: int, causal: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        # QKV projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # Scaling factor for dot products
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x):
        """x: (batch, seq, d_model) -> (batch, seq, d_model)"""
        batch, seq, d_model = x.shape

        # Project to Q, K, V: (batch, seq, d_model)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Reshape for multi-head: (batch, seq, n_heads, d_head) -> (batch, n_heads, seq, d_head)
        Q = Q.reshape(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        K = K.reshape(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        V = V.reshape(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: (batch, n_heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
            # Create causal mask: (seq, seq) with upper triangle = -inf
            mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over last dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values: (batch, n_heads, seq, d_head)
        attn_out = torch.matmul(attn_weights, V)

        # Reshape back: (batch, n_heads, seq, d_head) -> (batch, seq, n_heads, d_head) -> (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq, d_model)

        # Output projection
        output = self.Wo(attn_out)

        return output
