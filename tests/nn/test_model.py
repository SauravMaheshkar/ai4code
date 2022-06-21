"""Basic test to check model instantiation"""
import pytest
import torch
from torch import nn

from src.nn.model import MarkdownModel

from ..utils import ids_tensor, random_attention_mask


@pytest.mark.parametrize(
    ("model_name"),
    (
        ("microsoft/codebert-base"),
        ("microsoft/codebert-base-mlm"),
        ("huggingface/CodeBERTa-small-v1"),
    ),
)
def test_model_instantiation(model_name: str) -> None:
    """
    Test Model Instantiation

    :param model_name: Model Name
    :type model_name: str
    """

    model = MarkdownModel(model_name)

    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    ("model_name", "batch_size"),
    (
        ("microsoft/codebert-base", 16),
        ("microsoft/codebert-base-mlm", 16),
        ("huggingface/CodeBERTa-small-v1", 16),
    ),
)
def test_output_shape(
    model_name: str, batch_size: int, seq_length: int = 7, vocab_size: int = 99
) -> None:
    """Test to check output shape"""
    model = MarkdownModel(model_name)
    model.top = nn.Linear(784, 1)

    ids = ids_tensor(shape=[batch_size, seq_length], vocab_size=vocab_size)
    mask = random_attention_mask([batch_size, seq_length])
    fts = torch.rand(batch_size, batch_size)

    demo_input = (ids, mask, fts)
    output = model(*demo_input)

    assert isinstance(output, torch.Tensor)
    assert list(output.size()) == [batch_size, 1]
