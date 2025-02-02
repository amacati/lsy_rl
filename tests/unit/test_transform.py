import numpy as np
import pytest
import torch

from lsy_rl.core.transforms import ClipTF, IdentityTF, to_transforms


@pytest.mark.unit
def test_identity_tf():
    x = torch.rand(10, 10)
    identity_tf = IdentityTF()
    assert torch.allclose(identity_tf(x), x), "Identity transform should not change the input."
    x = np.random.rand(10, 10)
    assert np.allclose(identity_tf(x), x), "Identity transform should not change the input."


@pytest.mark.unit
def test_clip_tf():
    x = torch.linspace(-2, 2, 10)
    clip_tf = ClipTF(-1, 1)
    assert torch.allclose(clip_tf(x), torch.clamp(x, -1, 1)), (
        "Clip transform should clip the input."
    )


@pytest.mark.unit
def test_to_transforms():
    tf = to_transforms([{"type": "IdentityTF"}])
    assert isinstance(tf[0], IdentityTF), "Should return a list of IdentityTF."
    tf = to_transforms([{"type": "ClipTF", "kwargs": {"min": -0.7, "max": 1.1}}])
    assert isinstance(tf[0], ClipTF), "Should return a list of ClipTF."
    assert tf[0].params["min"] == -0.7, "Should set the min value of the ClipTF."
    assert tf[0].params["max"] == 1.1, "Should set the max value of the ClipTF."
