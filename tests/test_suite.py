"""20-test suite for the Transformers vs Diffusion study project.

The char tokenizer is implemented and fully tested here. Attention, model, and
sampling tests are skipped until the from-scratch modules are written.
"""
import numpy as np
import pytest
from src.common.tokenizer import CharTokenizer
from src.common.sdpa_reference import (
    softmax, scaled_dot_product_attention, multi_head_self_attention,
)


# --- tokenizer (pure, pass now) ----------------------------------------------
def test_vocab_size_counts_unique_chars():
    assert CharTokenizer("aabbc").vocab_size == 3

def test_encode_decode_roundtrip():
    t = CharTokenizer("hello world")
    assert t.decode(t.encode("hello")) == "hello"

def test_encode_returns_ints():
    t = CharTokenizer("abc")
    assert all(isinstance(i, int) for i in t.encode("abc"))

def test_chars_are_sorted():
    t = CharTokenizer("cba")
    assert t.chars == ["a", "b", "c"]

def test_stoi_itos_consistent():
    t = CharTokenizer("xyz")
    for c in "xyz":
        assert t.itos[t.stoi[c]] == c

def test_empty_then_known_char_encode():
    t = CharTokenizer("ab")
    assert t.encode("") == []

def test_vocab_size_single_char():
    assert CharTokenizer("aaaa").vocab_size == 1

def test_decode_full_vocab():
    t = CharTokenizer("ab")
    assert t.decode([t.stoi["a"], t.stoi["b"]]) == "ab"


# --- attention reference (implemented, pure NumPy) ---------------------------
def test_softmax_rows_sum_to_one():
    out = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(out.sum(axis=-1), 1.0)

def test_attention_output_shape_matches_input():
    x = np.random.rand(5, 8)
    out, _ = scaled_dot_product_attention(x, x, x)
    assert out.shape == (5, 8)

def test_attention_weights_are_row_stochastic():
    x = np.random.rand(4, 6)
    _, w = scaled_dot_product_attention(x, x, x)
    assert np.allclose(w.sum(axis=-1), 1.0)

def test_causal_mask_blocks_future_positions():
    x = np.random.rand(4, 6)
    _, w = scaled_dot_product_attention(x, x, x, causal=True)
    # upper triangle (future) must be zero
    assert np.allclose(np.triu(w, k=1), 0.0)

def test_bidirectional_attends_both_directions():
    x = np.random.rand(4, 6)
    _, w = scaled_dot_product_attention(x, x, x, causal=False)
    assert not np.allclose(np.triu(w, k=1), 0.0)

def test_multihead_output_shape():
    x = np.random.rand(5, 8)
    W = [np.random.rand(8, 8) for _ in range(4)]
    out = multi_head_self_attention(x, *W, n_heads=2)
    assert out.shape == (5, 8)

def test_head_dim_divides_d_model():
    x = np.random.rand(5, 8)
    W = [np.random.rand(8, 8) for _ in range(4)]
    with pytest.raises(ValueError):
        multi_head_self_attention(x, *W, n_heads=3)


# --- GPT (implement src/transformer/gpt.py) ----------------------------------
@pytest.mark.skip(reason="implement TinyGPT forward")
def test_gpt_logits_shape(): ...

@pytest.mark.skip(reason="implement TinyGPT.generate")
def test_gpt_generate_extends_sequence(): ...

@pytest.mark.skip(reason="train overfit sanity check")
def test_gpt_overfits_tiny_string(): ...


# --- masked / diffusion LM (implement src/diffusion/masked_lm.py) ------------
@pytest.mark.skip(reason="implement TinyMaskedLM forward")
def test_masked_lm_logits_shape(): ...

@pytest.mark.skip(reason="implement iterative unmasking")
def test_diffusion_generate_runs_n_steps(): ...

@pytest.mark.skip(reason="implement unmasking")
def test_diffusion_output_has_no_mask_tokens(): ...


# --- shared-foundation thesis ------------------------------------------------
@pytest.mark.skip(reason="both models reuse MultiHeadSelfAttention")
def test_both_models_share_attention_module(): ...
