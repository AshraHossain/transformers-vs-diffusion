"""Autoregressive GPT: causal blocks + left-to-right sampling.

TinyGPT implements an autoregressive language model with causal self-attention.
Each token can only attend to previous positions, enforcing left-to-right order.

Architecture:
    Token IDs → Embeddings → [TransformerBlock × n_layers] → Language Head → Logits

Each TransformerBlock contains:
    1. Layer norm + multi-head attention (causal=True from src.common.attention)
    2. Residual connection
    3. Layer norm + feedforward
    4. Residual connection

Training objective: Next-token prediction
    Given sequence [t0, t1, t2, ...], predict [t1, t2, t3, ...] in one forward pass.

Generation: Autoregressive sampling
    Given a prompt, repeatedly:
    1. Forward through model to get logits
    2. Sample next token from last position
    3. Append to sequence
    4. Repeat until max_new_tokens reached

Shared modules:
    - MultiHeadSelfAttention (src.common.attention) with causal=True
    - TokenPositionalEmbedding (src.common.embeddings)

Example:
    >>> gpt = TinyGPT(vocab_size=256, d_model=64, n_layers=2, n_heads=4)
    >>> token_ids = torch.tensor([[5, 3, 7, 2]])  # (batch=1, seq=4)
    >>> logits = gpt(token_ids)  # (batch=1, seq=4, vocab_size=256)
    >>> generated = gpt.generate(token_ids, max_new_tokens=10)  # (batch=1, seq=14)
"""
import torch
import torch.nn as nn
from src.common.embeddings import build_embeddings
from src.common.attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Single transformer block: causal attention + feedforward with residuals.

    Uses pre-normalization (norm before attention/feedforward) for better training
    stability. The causal attention mask enforces left-to-right information flow.

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

        # Causal self-attention
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True)

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


class TinyGPT(nn.Module):
    """Tiny autoregressive GPT: embeddings + causal transformer blocks + language head.

    A minimal GPT implementation demonstrating autoregressive language modeling.
    The model predicts the next token in a sequence given all previous tokens.

    All attention uses causal masking (from src.common.attention.MultiHeadSelfAttention
    with causal=True), preventing the model from attending to future positions.

    Args:
        vocab_size: Number of tokens in vocabulary.
        d_model: Embedding and hidden dimension (default 64).
        n_layers: Number of transformer blocks (default 2).
        n_heads: Number of attention heads per block (default 4).
                 Must divide d_model evenly.
        max_len: Maximum sequence length for positional encoding (default 512).

    Attributes:
        embed: TokenPositionalEmbedding (shared with masked-LM)
        blocks: List of TransformerBlock (each uses causal attention)
        norm: Final layer normalization
        head: Linear projection to vocab logits

    Shape:
        forward() input: (batch, seq) - token IDs in [0, vocab_size)
        forward() output: (batch, seq, vocab_size) - logits per position

        generate() input: (batch, init_seq) - starting token sequence
        generate() output: (batch, init_seq + max_new_tokens) - extended sequence

    Example:
        >>> model = TinyGPT(vocab_size=256, d_model=64, n_layers=2, n_heads=4)
        >>> token_ids = torch.tensor([[5, 3, 7]])
        >>> logits = model(token_ids)
        >>> logits.shape
        torch.Size([1, 3, 256])
        >>> generated = model.generate(token_ids, max_new_tokens=10)
        >>> generated.shape
        torch.Size([1, 13])

    Training:
        Optimize the cross-entropy loss between model logits and target tokens.
        Forward pass produces logits for all positions; compute loss on all
        positions except the first (which has no target).

        Example:
            >>> logits = model(token_ids)
            >>> loss = F.cross_entropy(
            ...     logits[:, :-1].reshape(-1, vocab_size),
            ...     token_ids[:, 1:].reshape(-1)
            ... )
    """

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

        # Embeddings (shared with masked-LM via build_embeddings)
        self.embed = build_embeddings(vocab_size, d_model, max_len)

        # Stack of causal transformer blocks
        # Each uses MultiHeadSelfAttention(causal=True)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        # Final layer norm (post-norm stabilizes training)
        self.norm = nn.LayerNorm(d_model)

        # Language modeling head: project d_model -> vocab logits
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        """token_ids: (batch, seq) -> logits: (batch, seq, vocab_size)"""
        # Embed tokens
        x = self.embed(token_ids)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Project to vocab logits
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        """Generate autoregressive sequence.

        Args:
            idx: (batch, seq) starting token indices
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature (>1 = more random, <1 = more deterministic)

        Returns:
            (batch, seq + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Get logits for all positions (but only use last token's logits for sampling)
            logits = self.forward(idx)
            logits = logits[:, -1, :] / temperature

            # Sample next token (greedy if temperature=1)
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

        return idx
