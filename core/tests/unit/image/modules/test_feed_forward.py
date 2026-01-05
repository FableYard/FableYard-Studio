import torch
from components.tranformers.modules.feed_forward import FeedForward


class TestFeedForwardModule:
    def test_init(self):
        dim = 512
        dim_out = None
        mult = 4
        dropout = 0.0
        activation_fn = "geglu"
        final_dropout = False
        inner_dim = None
        bias = True

        feed_forward = FeedForward(dim, dim_out, mult, dropout, activation_fn, final_dropout, inner_dim, bias)

        assert isinstance(feed_forward.net, torch.nn.ModuleList)
        assert len(feed_forward.net) == 3

        from components.tranformers.modules.activators.geglu import GEGLU

        assert isinstance(feed_forward.net[0], GEGLU)
        assert isinstance(feed_forward.net[1], torch.nn.Dropout)
        assert isinstance(feed_forward.net[2], torch.nn.Linear)

    def test_init_with_final_dropout(self):
        feed_forward = FeedForward(input_channel_count=512, final_dropout=True, dropout=0.1)

        assert len(feed_forward.net) == 4
        assert isinstance(feed_forward.net[3], torch.nn.Dropout)

    def test_init_different_activations(self):
        from components.tranformers.modules.activators.gelu import GELU
        from components.tranformers.modules.activators.swiglu import SwiGLU
        from components.tranformers.modules.activators.approximate_gelu import ApproximateGELU
        from components.tranformers.modules.activators import LinearActivation

        ff_gelu = FeedForward(input_channel_count=512, activation_function="gelu")
        ff_gelu_approximate = FeedForward(input_channel_count=512, activation_function="gelu-approximate")
        ff_swiglu = FeedForward(input_channel_count=512, activation_function="swiglu")
        ff_approx = FeedForward(input_channel_count=512, activation_function="geglu-approximate")
        ff_linear_silu = FeedForward(input_channel_count=512, activation_function="linear-silu")

        assert isinstance(ff_gelu.net[0], GELU)
        assert isinstance(ff_gelu_approximate.net[0], GELU)
        assert isinstance(ff_swiglu.net[0], SwiGLU)
        assert isinstance(ff_approx.net[0], ApproximateGELU)
        assert isinstance(ff_linear_silu.net[0], LinearActivation)

    def test_forward(self):
        dim = 512
        batch_size = 2
        seq_len = 16
        feed_forward = FeedForward(input_channel_count=dim)
        input_tensor = torch.randn(batch_size, seq_len, dim)

        output = feed_forward(input_tensor)

        assert output.shape == (batch_size, seq_len, dim)
        assert isinstance(output, torch.Tensor)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
