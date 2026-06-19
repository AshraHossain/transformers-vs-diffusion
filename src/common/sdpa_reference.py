"""Pure-NumPy reference implementation of scaled dot-product & multi-head
self-attention.

This is a *reference* you can check your from-scratch PyTorch implementation
(`src/common/attention.py`) against — same math, no framework, fast to test.
The whole point of Project 8 is that BOTH the GPT and the diffusion/masked LM
reuse this exact attention block; they differ only in the mask and the training
objective.
"""
from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, causal: bool = False):
    """q,k,v: (seq, d). Returns (out, weights) where out is (seq, d)."""
    q, k, v = np.asarray(q), np.asarray(k), np.asarray(v)
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)
    if causal:
        seq = scores.shape[0]
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def multi_head_self_attention(x, Wq, Wk, Wv, Wo, n_heads: int, causal: bool = False):
    """x: (seq, d_model). W*: (d_model, d_model). Returns (seq, d_model)."""
    x = np.asarray(x)
    seq, d_model = x.shape
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    d_head = d_model // n_heads
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    heads = []
    for h in range(n_heads):
        s = slice(h * d_head, (h + 1) * d_head)
        out, _ = scaled_dot_product_attention(q[:, s], k[:, s], v[:, s], causal=causal)
        heads.append(out)
    return np.concatenate(heads, axis=-1) @ Wo
