import torch
from image.transformers.flux.transformer import FluxTransformer2DModel


class TestFluxTransformer:
    def test_init(self):
        patch_size = 1
        in_channels = 64
        out_channels = None
        num_layers = 19
        num_single_layers = 38
        attention_head_dim = 128
        num_attention_heads = 24
        joint_attention_dim = 4096
        pooled_projection_dim = 768

        transformer = FluxTransformer2DModel(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        # Check stored attributes
        assert transformer.output_channel_count == in_channels
        assert transformer.inner_dim == num_attention_heads * attention_head_dim
        assert transformer.axes_dimensions_rope == [16, 56, 56]

        # Check transformer blocks
        assert len(transformer.transformer_blocks) == num_layers
        assert len(transformer.single_transformer_blocks) == num_single_layers

    def test_forward(self):
        # Use smaller dimensions for faster testing
        patch_size = 1
        in_channels = 64
        num_layers = 2
        num_single_layers = 2
        attention_head_dim = 128
        num_attention_heads = 4
        joint_attention_dim = 512
        pooled_projection_dim = 768

        transformer = FluxTransformer2DModel(
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        batch_size = 1
        image_seq_len = 16
        text_seq_len = 8

        # Create input tensors
        hidden_states = torch.randn(batch_size, image_seq_len, in_channels)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, joint_attention_dim)
        pooled_projections = torch.randn(batch_size, pooled_projection_dim)
        timestep = torch.tensor([1])
        img_ids = torch.randn(image_seq_len, 3)
        txt_ids = torch.randn(text_seq_len, 3)
        guidance = torch.tensor([3.5])

        # Forward pass
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )

        # Check output is a tuple with one element
        assert isinstance(output, tuple)
        assert len(output) == 1

        # Check output shape
        expected_channels = patch_size * patch_size * in_channels
        assert output[0].shape == (batch_size, image_seq_len, expected_channels)