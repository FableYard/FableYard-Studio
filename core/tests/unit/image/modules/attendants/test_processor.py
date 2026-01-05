"""
Tests for FluxAttnProcessor.
"""
from torch import randn

from components.tranformers.modules.attendants.attention import Attention
from components.tranformers.flux.modules.processor import AttentionProcessor


class TestFluxAttnProcessor:
    """Test suite for FluxAttnProcessor."""

    def test_call__basic(self):
        """Test basic processor call without encoder hidden states."""
        query_dim = 128
        heads = 4
        dim_head = 32
        batch_size, seq_len = 2, 16

        attn = Attention(query_dim, head_count=heads, dim_head=dim_head)
        processor = AttentionProcessor()

        hidden_states = randn(batch_size, seq_len, query_dim)

        output = processor(attn, hidden_states)

        assert output is not None
        assert output.numel() > 0

    def test_call__with_encoder(self):
        """Test processor with encoder hidden states (cross-attention)."""
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
        processor = AttentionProcessor()

        hidden_states = randn(batch_size, seq_len, query_dim)
        encoder_hidden_states = randn(batch_size, encoder_seq_len, added_kv_proj_dim)

        output, encoder_output = processor(
            attn,
            hidden_states,
            encoder_hidden_states
        )

        assert output.shape == (batch_size, seq_len, query_dim)
        assert encoder_output.shape == (batch_size, encoder_seq_len, query_dim)

    def test_call__with_rotary_embeddings(self):
        """Test processor with rotary position embeddings."""
        query_dim = 128
        heads = 4
        dim_head = 32
        batch_size, seq_len = 2, 16

        attn = Attention(query_dim, head_count=heads, dim_head=dim_head)
        processor = AttentionProcessor()

        hidden_states = randn(batch_size, seq_len, query_dim)

        # Create rotary embeddings (cos, sin tuple)
        # Shape should match the sequence and head_dim
        cos = randn(seq_len, dim_head)
        sin = randn(seq_len, dim_head)
        image_rotary_emb = (cos, sin)

        output = processor(
            attn,
            hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        assert output.shape == (batch_size, seq_len, query_dim)
