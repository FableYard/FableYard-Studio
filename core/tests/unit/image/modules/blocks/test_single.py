"""
Tests for FluxSingleTransformerBlock (single-stream block).
"""
from torch import randn

from components.tranformers.flux.modules.blocks.single import SingleStreamBlock


class TestFluxSingleTransformerBlock:
    """Test suite for FluxSingleTransformerBlock."""

    def test_initialization__basic(self):
        """Test basic initialization."""
        dim = 512
        num_attention_heads = 8
        attention_head_dim = 64

        block = SingleStreamBlock(dim, num_attention_heads, attention_head_dim)

        assert block.norm is not None
        assert block.proj_mlp is not None
        assert block.act_mlp is not None
        assert block.proj_out is not None
        assert block.attn is not None

    def test_forward__basic(self):
        """Test basic forward pass."""
        batch_size, seq_len, encoder_seq_len = 2, 16, 8
        dim = 512
        num_attention_heads = 8
        attention_head_dim = 64

        block = SingleStreamBlock(dim, num_attention_heads, attention_head_dim)

        hidden_states = randn(batch_size, seq_len, dim)
        encoder_hidden_states = randn(batch_size, encoder_seq_len, dim)
        guided_timesteps = randn(batch_size, dim)

        encoder_output, hidden_output = block(
            hidden_states, encoder_hidden_states, guided_timesteps
        )

        assert encoder_output is not None
        assert hidden_output is not None
        assert encoder_output.numel() > 0
        assert hidden_output.numel() > 0

    def test_forward__with_rotary_embeddings(self):
        """Test forward pass with rotary position embeddings."""
        batch_size, seq_len, encoder_seq_len = 2, 16, 8
        dim = 512
        num_attention_heads = 8
        attention_head_dim = 64

        block = SingleStreamBlock(dim, num_attention_heads, attention_head_dim)

        hidden_states = randn(batch_size, seq_len, dim)
        encoder_hidden_states = randn(batch_size, encoder_seq_len, dim)
        guided_timesteps = randn(batch_size, dim)

        # Create rotary embeddings
        total_seq_len = seq_len + encoder_seq_len
        cos = randn(total_seq_len, attention_head_dim)
        sin = randn(total_seq_len, attention_head_dim)
        image_rotary_emb = (cos, sin)

        encoder_output, hidden_output = block(
            hidden_states,
            encoder_hidden_states,
            guided_timesteps,
            image_rotary_emb=image_rotary_emb
        )

        assert encoder_output.shape == (batch_size, encoder_seq_len, dim)
        assert hidden_output.shape == (batch_size, seq_len, dim)
