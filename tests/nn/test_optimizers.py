"""Basic test to check 8-bit Optimizers"""
from __future__ import annotations

import torch


def test_adamw():
    """
    Simple test to check Installation and Instantiation
    """
    params = torch.nn.Parameter(torch.rand(10, 10))
    constant = torch.rand(10, 10)

    original = params.data.sum().item()

    adamw = torch.optim.AdamW([params])

    out = constant * params
    loss = out.sum()
    loss.backward()
    adamw.step()

    modified = params.data.sum().item()

    assert original != modified
