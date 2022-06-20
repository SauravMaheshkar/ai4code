"""Custom Model"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class MarkdownModel(nn.Module):
    """Custom Model Class"""

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.update(
            {
                "hidden_dropout_prob": 0.0,
                "gradient_checkpointing": True,
            }
        )
        self.model = AutoModel.from_config(self.config)
        self.top = nn.Linear(769, 1)

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, fts: torch.Tensor
    ) -> torch.Tensor:
        """Compute Forward Pass"""
        features = self.model(ids, mask)[0]
        features = torch.cat((features[:, 0, :], fts), 1)
        features = self.top(features)
        return features
