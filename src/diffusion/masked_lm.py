"""Diffusion/masked LM: bidirectional blocks + iterative unmasking.

TinyMaskedLM implements a masked language model with bidirectional self-attention.
Unlike GPT (which predicts left-to-right), this model can attend to any position,
including future positions.

Architecture:
    Masked Token IDs → Embeddings → [TransformerBlock × n_layers] → Denoising Head → Logits

Each TransformerBlock contains:
    1. Layer norm + multi-head attention (causal=False from src.common.attention)
    2. Residual connection
    3. Layer norm + feedforward
    4. Residual connection

Training objective: Masked language modeling (denoising)
    Given a sequence with some positions masked:
    1. Forward through model to predict logits
    2. Compute loss only on masked positions (predicting the original token)
    3. Gradient updates only affect masked position predictions

Generation: Iterative unmasking (diffusion-style)
    Start with all tokens masked, then progressively unmask confident predictions:
    1. Initialize sequence with all MASK tokens (ID 0)
    2. For each iteration:
        a. Forward through model to get logits
        b. Calculate prediction confidence for masked positions
        c. Unmask the most confident ~1/steps predictions
        d. Repeat until all unmasked
    Result: A gradually refined sequence through multiple denoising steps

Shared modules:
    - MultiHeadSelfAttention (src.common.attention) with causal=False
    - TokenPositionalEmbedding (src.common.embeddings)

The key difference from GPT:
    - Bidirectional attention (no causal mask)
    - Iterative generation instead of autoregressive

Example:
    >>> masked_lm = TinyMaskedLM(vocab_size=256, d_model=64, n_layers=2, n_heads=4)
    >>> token_ids = torch.tensor([[0, 5, 0, 7, 0]])  # Some masked (0), some known
    >>> logits = masked_lm(token_ids)  # (batch=1, seq=5, vocab_size=256)
    >>> generated = masked_lm.generate(length=10, steps=4)  # (batch=1, seq=10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.embeddings import build_embeddings
from src.common.attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Single transformer block: bidirectional attention + feedforward.

    Uses pre-normalization. The attention is bidirectional (no causal mask),
    allowing each position to attend to all other positions in the sequence.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        ff_dim: Feedforward hidden dimension (default: 4 * d_model).

    Shape:
        Input: (batch, seq, d_model)
        Output: (batch, seq, d_model)

    Example:
        >>> block = TransformerBlock(d_model=64, n_heads=4)
        >>> x = torch.randn(2, 10, 64)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 10, 64])
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int = None):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Bidirectional self-attention (causal=False)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=False)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x):
        """x: (batch, seq, d_model) -> (batch, seq, d_model)"""
        # Attention with residual: pre-norm
        x = x + self.attn(self.norm1(x))

        # Feedforward with residual: pre-norm
        x = x + self.ff(self.norm2(x))

        return x


class TinyMaskedLM(nn.Module):
    """Tiny masked LM: bidirectional transformer + denoising objective.

    A minimal masked language model demonstrating denoising and iterative generation.
    The model predicts masked tokens given both left and right context.

    All attention is bidirectional (from src.common.attention.MultiHeadSelfAttention
    with causal=False), allowing information flow in both directions.

    Args:
        vocab_size: Number of tokens in vocabulary (0 reserved for MASK token).
        d_model: Embedding and hidden dimension (default 64).
        n_layers: Number of transformer blocks (default 2).
        n_heads: Number of attention heads per block (default 4).
                 Must divide d_model evenly.
        max_len: Maximum sequence length for positional encoding (default 512).

    Attributes:
        MASK_TOKEN_ID: Special token reserved for masking (always 0).
        embed: TokenPositionalEmbedding (shared with GPT)
        blocks: List of TransformerBlock (each uses bidirectional attention)
        norm: Final layer normalization
        head: Linear projection to vocab logits

    Shape:
        forward() input: (batch, seq) - token IDs in [0, vocab_size)
                         Position with value 0 (MASK) indicates unknown token
        forward() output: (batch, seq, vocab_size) - logits per position

        generate() input: None (generates from scratch)
        generate() output: (batch=1, length) - generated token sequence

    Example:
        >>> model = TinyMaskedLM(vocab_size=256, d_model=64, n_layers=2, n_heads=4)
        >>> # Create input with some positions masked
        >>> token_ids = torch.tensor([[5, 0, 7, 0, 3]])  # 0s are masked
        >>> logits = model(token_ids)
        >>> logits.shape
        torch.Size([1, 5, 256])
        >>> # Generate by iterative unmasking
        >>> generated = model.generate(length=10, steps=4)
        >>> generated.shape
        torch.Size([1, 10])

    Training:
        1. Create batches with random masking (e.g., mask 50% of tokens)
        2. Forward through model to get logits
        3. Compute loss only on masked positions (cross-entropy with original tokens)
        4. Backprop and update

        Example:
            >>> batch = torch.randint(1, 256, (4, 20))  # Vocab 1-255 (0 reserved)
            >>> masked_batch = batch.clone()
            >>> mask = torch.rand_like(batch, dtype=torch.float) < 0.5
            >>> masked_batch[mask] = 0  # Mask random positions
            >>> logits = model(masked_batch)
            >>> loss = F.cross_entropy(
            ...     logits[mask].reshape(-1, 256),
            ...     batch[mask].reshape(-1)
            ... )

    Generation (iterative unmasking):
        1. Start with sequence of all MASK tokens
        2. Iteratively:
            a. Forward to get predictions for masked positions
            b. Unmask the most confident predictions (~1/steps per iteration)
            c. Repeat until no masks remain
        3. Result: Sequence refined over multiple denoising steps

    This demonstrates the diffusion paradigm: noisy input → denoising → clean output.
    """

    MASK_TOKEN_ID = 0  # Reserved for masking

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embeddings (shared design with GPT)
        self.embed = build_embeddings(vocab_size, d_model, max_len)

        # Stack of bidirectional transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Denoising head: project d_model -> vocab logits
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        """token_ids: (batch, seq) -> logits: (batch, seq, vocab_size)"""
        # Embed tokens (including masked positions)
        x = self.embed(token_ids)

        # Pass through bidirectional transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Project to vocab logits
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(self, length: int, steps: int = 8, temperature: float = 1.0):
        """Iteratively unmask sequences via denoising.

        Args:
            length: sequence length to generate
            steps: number of denoising iterations
            temperature: sampling temperature

        Returns:
            (batch=1, length) token sequence
        """
        device = next(self.parameters()).device

        # Start with all tokens masked
        token_ids = torch.full((1, length), self.MASK_TOKEN_ID, device=device, dtype=torch.long)
        mask_positions = torch.ones((1, length), dtype=torch.bool, device=device)

        # Iterative unmasking
        for step in range(steps):
            # Forward pass: predict logits at all positions
            logits = self.forward(token_ids)  # (1, length, vocab_size)

            # Get logits for masked positions
            mask_pos_indices = torch.where(mask_positions[0])[0]
            logits_masked = logits[0, mask_pos_indices, :]  # (n_masked, vocab_size)

            # Calculate confidence: max probability for each position
            probs_masked = F.softmax(logits_masked / temperature, dim=-1)
            confidence, predicted_tokens = probs_masked.max(dim=-1)

            # Decide which positions to unmask this iteration
            # Unmask approximately 1/steps of remaining masks each iteration
            n_unmask = max(1, (mask_positions.sum().item()) // (steps - step))
            n_unmask = min(n_unmask, len(mask_pos_indices))  # Don't unmask more than exist

            # Get most confident predictions
            _, confidence_order = torch.topk(confidence, k=n_unmask, largest=True)
            positions_to_unmask = mask_pos_indices[confidence_order]

            # Update tokens at unmasked positions
            for pos in positions_to_unmask:
                pos_idx = (mask_pos_indices == pos).nonzero(as_tuple=True)[0].item()
                token_ids[0, pos] = predicted_tokens[pos_idx]
                mask_positions[0, pos] = False

            # If no masks left, we're done
            if not mask_positions.any():
                break

        return token_ids
