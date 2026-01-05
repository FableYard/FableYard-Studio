"""
Tests for attendants utility functions.
"""
from torch import randn

from components.tranformers.modules.attendants.attention import Attention
from components.tranformers.modules.attendants.utils import _get_qkv_projections


class TestGetQKVProjections:
    """Test suite for _get_qkv_projections utility function."""

    def test_basic_projections(self):
        """Test basic Q, K, V projections without encoder hidden states."""
        query_dim = 128
        heads = 4
        dim_head = 32
        batch_size, seq_len = 2, 16

        attn = Attention(query_dim, head_count=heads, dim_head=dim_head)
        hidden_states = randn(batch_size, seq_len, query_dim)

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states
        )

        # Check basic projections exist and have correct shape
        inner_dim = heads * dim_head
        assert query.shape == (batch_size, seq_len, inner_dim)
        assert key.shape == (batch_size, seq_len, inner_dim)
        assert value.shape == (batch_size, seq_len, inner_dim)

        # Check encoder projections are None
        assert encoder_query is None
        assert encoder_key is None
        assert encoder_value is None

    def test_projections_with_encoder(self):
        """Test Q, K, V projections with encoder hidden states."""
        query_dim = 128
        heads = 4
        dim_head = 32
        added_kv_proj_dim = 256
        batch_size, seq_len, encoder_seq_len = 2, 16, 8

        attn = Attention(
            query_dim,
            head_count=heads,
            dim_head=dim_head,
            added_kv_proj_dim=added_kv_proj_dim
        )
        hidden_states = randn(batch_size, seq_len, query_dim)
        encoder_hidden_states = randn(batch_size, encoder_seq_len, added_kv_proj_dim)

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        # Check main projections
        inner_dim = heads * dim_head
        assert query.shape == (batch_size, seq_len, inner_dim)
        assert key.shape == (batch_size, seq_len, inner_dim)
        assert value.shape == (batch_size, seq_len, inner_dim)

        # Check encoder projections exist and have correct shape
        assert encoder_query is not None
        assert encoder_key is not None
        assert encoder_value is not None
        assert encoder_query.shape == (batch_size, encoder_seq_len, inner_dim)
        assert encoder_key.shape == (batch_size, encoder_seq_len, inner_dim)
        assert encoder_value.shape == (batch_size, encoder_seq_len, inner_dim)
