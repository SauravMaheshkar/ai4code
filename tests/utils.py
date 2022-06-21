"""Utility Fuctions for Testing"""
import random

import torch

global_rng = random.Random()


def ids_tensor(shape, vocab_size):
    """
    Create a Tensor for ids
    """
    rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return (
        torch.tensor(data=values, dtype=torch.long, device="cpu")
        .view(shape)
        .contiguous()
    )


def random_attention_mask(shape):
    """Create Attention Mask"""
    attn_mask = ids_tensor(shape, vocab_size=2)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask
