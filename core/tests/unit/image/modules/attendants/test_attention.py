"""
Tests for FluxAttention module.
"""
from torch import randn

from components.tranformers.modules.attendants.attention import Attention
from components.tranformers.flux.modules.processor import AttentionProcessor


class TestFluxAttention:
    """Test suite for FluxAttention module."""

    def test_initialization__basic(self):
        """Test basic initialization of FluxAttention."""
        query_dim = 128
        heads = 4
        dim_head = 32

        attn = Attention(query_dim, head_count=heads, dim_head=dim_head)

        assert attn.query_dim == query_dim
        assert attn.heads == heads
        assert attn.head_dim == dim_head
        assert attn.inner_dim == heads * dim_head
        assert attn.to_q is not None
        assert attn.to_k is not None
        assert attn.to_v is not None
        assert attn.norm_q is not None
        assert attn.norm_k is not None

    def test_initialization__with_added_kv(self):
        """Test initialization with added key-value projections."""
        query_dim = 128
        heads = 4
        dim_head = 32
        added_kv_proj_dim = 256

        attn = Attention(
            query_dim,
            head_count=heads,
            dim_head=dim_head,
            added_kv_proj_dim=added_kv_proj_dim
        )

        assert attn.added_kv_proj_dim == added_kv_proj_dim
        assert attn.add_q_proj is not None
        assert attn.add_k_proj is not None
        assert attn.add_v_proj is not None
        assert attn.to_add_out is not None
        assert attn.norm_added_q is not None
        assert attn.norm_added_k is not None

    def test_forward__basic(self):
        """Test basic forward pass without encoder hidden states."""
        query_dim = 128
        heads = 4
        dim_head = 32
        batch_size, seq_len = 2, 16

        processor = AttentionProcessor()
        attn = Attention(
            query_dim,
            head_count=heads,
            dim_head=dim_head,
            processor=processor
        )

        hidden_states = randn(batch_size, seq_len, query_dim)

        output = attn(hidden_states)

        assert output is not None
        assert output.numel() > 0
