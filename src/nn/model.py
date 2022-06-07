"""Custom Model"""
import torch
from torch import nn
from transformers import AutoModel


class MarkdownModel(nn.Module):
    """Custom Model Class"""

    def __init__(self, model_path: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, fts: torch.Tensor
    ) -> torch.Tensor:
        """Compute Forward Pass"""
        features = self.model(ids, mask)[0]
        features = torch.cat((features[:, 0, :], fts), 1)
        features = self.top(features)
        return features
