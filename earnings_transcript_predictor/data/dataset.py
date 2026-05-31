"""
dataset.py

PyTorch Dataset class that pairs transcript text with post-earnings
return labels. Designed to feed into a FinBERT-based model.

Model inputs:  tokenized transcript text  (input_ids, attention_mask)
Target:        return_pct (float) — post-earnings percentage return
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EarningsDataset(Dataset):
    """
    PyTorch Dataset for earnings call transcripts + return labels.

    Args:
        records (list[dict]): Each dict must have:
            - "transcript" (str): full transcript text
            - "return_pct" (float): the target return (e.g. 2.3 or -1.7)
        tokenizer_name (str): HuggingFace model name for tokenization.
                              Default: "ProsusAI/finbert"
        max_length (int): Max token length. FinBERT supports up to 512.
                          Earnings calls are long; we truncate to 512.
    """

    def __init__(
        self,
        records: list,
        tokenizer_name: str = "ProsusAI/finbert",
        max_length: int = 512,
    ):
        self.records = records
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record["transcript"]
        label = float(record["return_pct"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            # squeeze removes the batch dim the tokenizer adds
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
            # keep metadata for debugging / analysis
            "symbol": record.get("symbol", ""),
            "quarter": record.get("quarter", ""),
            "earnings_date": record.get("earnings_date", ""),
        }
