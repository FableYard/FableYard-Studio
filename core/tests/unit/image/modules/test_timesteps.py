import torch
from components.tranformers.modules.timesteps import Timesteps


class TestTimesteps:
    def test_init(self):
        num_channels = 256
        flip_sin_to_cos = True
        downscale_freq_shift = 1.0
        scale = 2

        timesteps = Timesteps(num_channels, flip_sin_to_cos, downscale_freq_shift, scale)

        assert timesteps.channel_count == num_channels
        assert timesteps.flip_sin_to_cos == flip_sin_to_cos
        assert timesteps.downscale_freq_shift == downscale_freq_shift
        assert timesteps.scale == scale

    def test_forward(self):
        num_channels = 256
        batch_size = 4

        timesteps_module = Timesteps(num_channels, True, 1.0, 1)
        timesteps_input = torch.randn(batch_size)

        output = timesteps_module(timesteps_input)

        assert output.shape == (batch_size, num_channels)