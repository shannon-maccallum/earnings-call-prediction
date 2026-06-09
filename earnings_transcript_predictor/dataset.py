"""
PyTorch dataset for earnings call transcripts and post-earnings returns.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def flatten_transcript(data: dict) -> str:
    """Join Alpha Vantage speaker turns into one transcript string."""
    transcript = data["transcript"]
    if isinstance(transcript, list):
        return " ".join(turn.get("content", "") for turn in transcript)
    return str(transcript)


def load_records(data_dir: str) -> list:
    """Load transcript JSON files that contain both text and a return label."""
    records = []
    for path in sorted(Path(data_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if "transcript" in data and "return_pct" in data:
            data = dict(data)
            data["transcript"] = flatten_transcript(data)
            records.append(data)
    print(f"Loaded {len(records)} transcripts from {data_dir}")
    return records


class EarningsDataset(Dataset):
    """Tokenized transcript dataset for FinBERT regression."""

    def __init__(
        self,
        records: list,
        tokenizer_name: str = "ProsusAI/finbert",
        max_length: int = 512,
    ):
        self.records = records
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        encoding = self.tokenizer(
            record["transcript"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(record["return_pct"], dtype=torch.float),
            "symbol": record.get("symbol", ""),
            "quarter": record.get("quarter", ""),
        }
