"""FinBERT regression model for predicting post-earnings stock returns."""

import torch
import torch.nn as nn
from transformers import AutoModel


class EarningsModel(nn.Module):
    """FinBERT encoder with a single-output regression head."""

    def __init__(self, model_name: str = "ProsusAI/finbert", dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.regressor(self.dropout(cls)).squeeze(-1)

    def freeze_bert(self) -> None:
        """Freeze FinBERT parameters so only the regression head trains."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self) -> None:
        """Unfreeze FinBERT parameters for full fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True

