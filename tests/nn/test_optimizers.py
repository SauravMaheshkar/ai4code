"""Basic test to check 8-bit Optimizers"""
from __future__ import annotations

import bitsandbytes as bnb
import torch


def test_adamw8bit():
    """
    Simple test to check Installation and Instantiation
    """
    params = torch.nn.Parameter(torch.rand(10, 10).cuda())
    constant = torch.rand(10, 10).cuda()

    original = params.data.sum().item()

    adamw = bnb.optim.AdamW8bit([params])

    out = constant * params
    loss = out.sum()
    loss.backward()
    adamw.step()

    modified = params.data.sum().item()

    assert original != modified
