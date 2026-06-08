"""
models.py

FinBERT-based regression model for predicting post-earnings stock returns.

Architecture:
    - Backbone: ProsusAI/finbert (BERT pre-trained on 4.9B tokens of financial text)
    - Head:     Linear(768 -> 1) regression layer
    - Output:   Predicted return_pct (continuous float, e.g. 2.45 or -3.71)

Training uses gradual unfreezing (Howard & Ruder, 2018):
    Phase 1: Freeze FinBERT, train only the Linear head (lr=1e-3, 2 epochs)
    Phase 2: Unfreeze all, fine-tune full model (lr=2e-5, 3 epochs)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class EarningsModel(nn.Module):
    """
    Fine-tuned FinBERT for earnings call return regression.

    Replaces FinBERT's original 3-class sentiment output with a single
    Linear(768->1) layer that predicts a continuous post-earnings return %.

    Args:
        model_name: HuggingFace model identifier. Default: ProsusAI/finbert
        dropout:    Dropout rate on the regression head. Default: 0.1
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids:      (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) 1=real token, 0=padding

        Returns:
            (batch_size,) predicted return percentages
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        cls = self.dropout(cls)
        return self.regressor(cls).squeeze(-1)

    def freeze_bert(self):
        """Freeze all FinBERT parameters (Phase 1 of gradual unfreezing)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        """Unfreeze all FinBERT parameters (Phase 2 of gradual unfreezing)."""
        for param in self.bert.parameters():
            param.requires_grad = True
