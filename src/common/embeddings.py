"""Token + positional embeddings for both GPT and masked-LM.

This module provides the embedding layer shared by both generation paradigms.
It combines learned token embeddings with sinusoidal positional encoding to give
each position in a sequence a unique representation.

Token embedding: Maps discrete token IDs to d_model-dimensional vectors via
a learnable embedding table (vocab_size x d_model).

Positional encoding: Sinusoidal functions at different frequencies encode absolute
positions without requiring training. Avoids the need for learnable position embeddings
and generalizes to sequences longer than those seen during training.

The sinusoidal encoding uses:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Example usage:
    >>> embed = build_embeddings(vocab_size=256, d_model=64, max_len=512)
    >>> token_ids = torch.tensor([[5, 3, 7, 2]])  # (batch=1, seq=4)
    >>> embedded = embed(token_ids)
    >>> embedded.shape
    torch.Size([1, 4, 64])

This embedding layer is used identically by both GPT and masked-LM.
"""
import torch
import torch.nn as nn
import math


class TokenPositionalEmbedding(nn.Module):
    """Combines token embeddings with sinusoidal positional encoding.

    Learns a token embedding table and registers fixed sinusoidal positional
    encodings. Each token is embedded, then the position embedding is added.

    Args:
        vocab_size: Number of tokens in vocabulary.
        d_model: Embedding dimension.
        max_len: Maximum sequence length (determines PE table size).

    Shape:
        Input: (batch, seq) - token IDs, each in [0, vocab_size)
        Output: (batch, seq, d_model) - embedded tokens with positions

    Example:
        >>> embed = TokenPositionalEmbedding(vocab_size=100, d_model=32, max_len=200)
        >>> token_ids = torch.randint(0, 100, (4, 20))
        >>> out = embed(token_ids)
        >>> out.shape
        torch.Size([4, 20, 32])
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.d_model = d_model

        # Token embedding table
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not trainable, but part of model state)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, token_ids):
        """token_ids: (batch, seq) -> (batch, seq, d_model)"""
        batch, seq = token_ids.shape

        # Token embeddings
        x = self.token_embed(token_ids)

        # Add positional encoding
        x = x + self.pe[:, :seq, :]

        return x


def build_embeddings(vocab_size: int, d_model: int, max_len: int):
    """Factory function returning an embedding layer.

    Args:
        vocab_size: number of tokens in vocabulary
        d_model: embedding dimension
        max_len: maximum sequence length

    Returns:
        TokenPositionalEmbedding module that maps (batch, seq) token IDs
        to (batch, seq, d_model) embedded vectors.
    """
    return TokenPositionalEmbedding(vocab_size, d_model, max_len)
