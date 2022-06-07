"""Optimizer Utilites"""
from typing import List

import torch
from torch import nn
from torch.optim import Optimizer


def fetch_optimizer(model: nn.Module, weight_decay: float) -> Optimizer:
    """
    Create Optimizer

    :param model: Model
    :type model: nn.Module
    :param weight_decay: Weight Decay Value
    :type weight_decay: float
    :return: Optimizer
    :rtype: torch.optim.Optimizer
    """
    param_optimizer: List = list(model.named_parameters())
    no_decay: List = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)

    return optimizer
