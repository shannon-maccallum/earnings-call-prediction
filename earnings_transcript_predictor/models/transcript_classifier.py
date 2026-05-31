"""
transcript_classifier.py

FinBERT-based regression model for predicting post-earnings stock returns.

Architecture:
    - Backbone: ProsusAI/finbert (BERT pre-trained on financial text)
    - Head:     Linear(768 -> 1) regression head
    - Output:   Predicted return_pct (unbounded float)

NOTE: This is a model stub for Milestone 2. Training code will be added
in subsequent milestones.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TranscriptReturnPredictor(nn.Module):
    """
    Fine-tuned FinBERT for earnings call return regression.

    Args:
        model_name (str): HuggingFace model identifier.
        dropout (float):  Dropout rate on the regression head.
        freeze_backbone (bool): If True, freeze FinBERT weights and only
                                train the regression head. Useful for
                                quick baselines.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # 768 for BERT-base
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation as the sequence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        prediction = self.regressor(cls_embedding).squeeze(-1)
        return prediction  # shape: (batch_size,)
