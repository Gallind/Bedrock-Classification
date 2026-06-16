"""unet.py: shapes, channel-count flexibility, input-size validation."""

import pytest
import torch

from seabed_unet.unet import UNet


@pytest.mark.parametrize("in_channels", [2, 3])
def test_output_shape(in_channels):
    model = UNet(in_channels=in_channels, num_classes=3, base_filters=8, depth=2)
    x = torch.randn(2, in_channels, 16, 16)
    assert model(x).shape == (2, 3, 16, 16)


def test_real_geometry_forward():
    model = UNet(in_channels=3, num_classes=3, base_filters=16, depth=4)
    x = torch.randn(1, 3, 128, 128)
    assert model(x).shape == (1, 3, 128, 128)


def test_param_count_is_lightweight():
    model = UNet(in_channels=3, num_classes=3, base_filters=16, depth=4)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 3_000_000, f"{n_params:,} params is no longer 'lightweight'"


def test_indivisible_input_raises():
    model = UNet(in_channels=3, num_classes=3, base_filters=8, depth=4)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 100, 100))


def test_gradients_flow_to_all_parameters():
    model = UNet(in_channels=2, num_classes=3, base_filters=8, depth=2)
    out = model(torch.randn(1, 2, 16, 16))
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
