"""
dataset.py

PyTorch Dataset for earnings call transcripts paired with post-earnings returns.

Each record is a JSON file with:
    - transcript: list of speaker turns (each with 'content' key) or a plain string
    - return_pct:  float, 3-day post-earnings return percentage (target variable)
    - symbol:      ticker symbol (e.g. "AAPL")
    - quarter:     quarter string (e.g. "2024Q1")
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def flatten_transcript(data: dict) -> str:
    """
    Join all speaker turns into a single string.
    Handles both list-of-turns format (Alpha Vantage API) and plain string.
    """
    if isinstance(data["transcript"], list):
        return " ".join([turn["content"] for turn in data["transcript"]])
    return data["transcript"]


def load_records(data_dir: str) -> list:
    """
    Load all transcript JSON files that have both 'transcript' and 'return_pct' keys.

    Args:
        data_dir: Path to directory containing .json transcript files.

    Returns:
        List of dicts, each with transcript text and return_pct label.
    """
    records = []
    for path in Path(data_dir).glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        if "transcript" in data and "return_pct" in data:
            data["transcript"] = flatten_transcript(data)
            records.append(data)
    print(f"Loaded {len(records)} transcripts from {data_dir}")
    return records


class EarningsDataset(Dataset):
    """
    PyTorch Dataset pairing tokenized transcripts with return labels.

    Model inputs:
        input_ids:      (seq_len,) tensor of token IDs
        attention_mask: (seq_len,) tensor, 1=real token, 0=padding

    Target:
        label: scalar float, post-earnings return % (e.g. 2.45 or -3.71)

    Args:
        records:        List of dicts from load_records()
        tokenizer_name: HuggingFace tokenizer. Default: ProsusAI/finbert
        max_length:     Max token sequence length. Default: 512
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
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(record["return_pct"], dtype=torch.float),
            "symbol":         record.get("symbol", ""),
            "quarter":        record.get("quarter", ""),
        }
