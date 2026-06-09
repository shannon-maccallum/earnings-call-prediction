"""Dataset and data-loading helpers for the earnings call project."""

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import torch
import yfinance as yf
from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def clean_transcript(text: str) -> str:
    """Normalize whitespace in transcript text."""
    return re.sub(r"\s+", " ", str(text)).strip()


def flatten_transcript(data: dict) -> str:
    """Join Alpha Vantage speaker turns into one transcript string."""
    transcript = data["transcript"]
    if isinstance(transcript, list):
        return clean_transcript(" ".join(turn.get("content", "") for turn in transcript))
    return clean_transcript(transcript)


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


class TranscriptLoader:
    """Load Alpha Vantage earnings call transcripts with local JSON caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "notebooks/data/transcripts",
        sleep_seconds: float = 12.0,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds

    def load(self, symbol: str, quarter: str) -> dict:
        path = self.cache_dir / f"{symbol}_{quarter}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

        if not self.api_key:
            raise ValueError("Set ALPHA_VANTAGE_API_KEY in .env to fetch new transcripts.")

        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": symbol,
                "quarter": quarter,
                "apikey": self.api_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if "transcript" not in data:
            raise ValueError(f"No transcript returned for {symbol} {quarter}: {data}")

        data.setdefault("symbol", symbol)
        data.setdefault("quarter", quarter)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        time.sleep(self.sleep_seconds)
        return data


class PriceLoader:
    """Fetch post-earnings returns from Yahoo Finance."""

    def __init__(self, return_window: int = 3):
        self.return_window = return_window

    def get_return(self, symbol: str, earnings_date: str) -> dict:
        start = datetime.strptime(earnings_date, "%Y-%m-%d")
        end = start + timedelta(days=self.return_window + 10)
        prices = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
        if len(prices) <= self.return_window:
            raise ValueError(f"Not enough price rows for {symbol} after {earnings_date}")

        price_day0 = float(prices["Close"].iloc[0])
        price_dayn = float(prices["Close"].iloc[self.return_window])
        return_pct = (price_dayn - price_day0) / price_day0 * 100
        return {
            "symbol": symbol,
            "earnings_date": earnings_date,
            "price_day0": price_day0,
            f"price_day{self.return_window}": price_dayn,
            "return_pct": return_pct,
        }

    def get_returns_batch(self, pairs: List[Tuple[str, str]]) -> List[dict]:
        results = []
        for symbol, earnings_date in pairs:
            try:
                results.append(self.get_return(symbol, earnings_date))
            except Exception as exc:
                results.append(
                    {
                        "symbol": symbol,
                        "earnings_date": earnings_date,
                        "error": str(exc),
                    }
                )
        return results

