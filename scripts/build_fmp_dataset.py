#!/usr/bin/env python3
"""Build model-ready earnings-call records from the FMP API.

This script fetches earnings-call transcripts from Financial Modeling Prep,
labels each call with the stock's 3-trading-day post-call return using FMP
historical prices, and writes JSON records in the format expected by
train_models.py:

    {
      "symbol": "AAPL",
      "quarter": "2024Q1",
      "earnings_date": "2024-02-01",
      "transcript": "...",
      "return_pct": 1.23
    }

Set FMP_API_KEY in your environment or .env file before running.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_TICKERS = [
    # Mega-cap tech / comms
    "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "AMZN", "TSLA", "NFLX", "ADBE",
    "CRM", "ORCL", "INTC", "AMD", "QCOM", "AVGO", "CSCO", "IBM", "TXN", "MU",
    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "V", "MA", "AXP", "BLK",
    "SCHW", "USB", "PNC", "TFC", "COF", "MET", "PRU", "AIG", "ICE", "CME",
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "BMY", "GILD", "AMGN", "TMO",
    "DHR", "ABT", "MDT", "CVS", "CI", "HUM", "ISRG", "REGN", "VRTX", "ZTS",
    # Consumer / retail / restaurants
    "WMT", "COST", "TGT", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS", "CMCSA",
    "PEP", "KO", "PG", "CL", "KMB", "MDLZ", "EL", "LULU", "TJX", "BKNG",
    # Industrials / transport / aerospace
    "BA", "CAT", "GE", "HON", "UPS", "FDX", "LMT", "RTX", "DE", "MMM",
    "UNP", "NSC", "CSX", "ETN", "EMR", "ITW", "WM", "GD", "NOC", "PH",
    # Energy / materials / utilities
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO", "HAL",
    "LIN", "APD", "SHW", "FCX", "NEM", "DOW", "NEE", "DUK", "SO", "AEP",
    # More liquid U.S. companies
    "NOW", "SHOP", "UBER", "ABNB", "PYPL", "SQ", "SNOW", "PLTR", "PANW", "CRWD",
    "ADP", "INTU", "ADI", "LRCX", "KLAC", "AMAT", "MRVL", "MCHP", "SNPS", "CDNS",
    "SPGI", "MCO", "CB", "MMC", "AON", "TRV", "ALL", "AFL", "BK", "STT",
    "SYK", "BSX", "BDX", "EW", "HCA", "MCK", "CAH", "IQV", "DXCM", "IDXX",
    "ROST", "DG", "DLTR", "YUM", "CMG", "MAR", "HLT", "RCL", "CCL", "GM",
    "F", "TMUS", "T", "VZ", "CHTR", "EA", "TTWO", "PARA", "FOXA", "NWSA",
    "ROK", "CMI", "PCAR", "URI", "HWM", "TDG", "JCI", "CARR", "OTIS", "IR",
    "KMI", "WMB", "OKE", "BKR", "DVN", "FANG", "HES", "KDP", "MNST",
    "EXC", "SRE", "XEL", "PEG", "ED", "EIX", "WEC", "D", "PPL", "AWK",
]


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip().upper() for item in items if item.strip()))


def load_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return dedupe(args.tickers.split(","))
    if args.ticker_file:
        with args.ticker_file.open() as f:
            return dedupe([line.strip() for line in f if line.strip() and not line.startswith("#")])
    return dedupe(DEFAULT_TICKERS)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


class FMPClient:
    def __init__(self, api_key: str, sleep_seconds: float = 0.1):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds

    def get(self, endpoint: str, params: dict[str, Any]) -> Any:
        params = dict(params)
        params["apikey"] = self.api_key
        response = requests.get(
            f"https://financialmodelingprep.com/stable/{endpoint}",
            params=params,
            timeout=30,
        )
        time.sleep(self.sleep_seconds)
        if response.status_code != 200:
            safe_text = response.text.replace(self.api_key, "HIDDEN")
            raise ValueError(f"fmp_http_{response.status_code}: {safe_text[:300]}")
        data = response.json()
        if isinstance(data, dict) and "Error Message" in data:
            raise ValueError(f"fmp_error: {data['Error Message']}")
        return data

    def transcript(self, symbol: str, year: int, quarter: int) -> dict[str, Any] | None:
        data = self.get(
            "earning-call-transcript",
            {"symbol": symbol, "year": year, "quarter": quarter},
        )
        if not isinstance(data, list) or not data:
            return None
        return data[0]

    def prices(self, symbol: str, start_date: str, days_ahead: int = 14) -> list[dict[str, Any]]:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = start + timedelta(days=days_ahead)
        data = self.get(
            "historical-price-eod/full",
            {
                "symbol": symbol,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
            },
        )
        if not isinstance(data, list) or not data:
            raise ValueError(f"no_price_data: {symbol} {start_date}")
        return data


def quarter_from_period(period: Any) -> int | None:
    text = str(period or "").upper().replace("Q", "").strip()
    if text in {"1", "2", "3", "4"}:
        return int(text)
    return None


def validate_transcript_date(item: dict[str, Any], requested_year: int, max_year_gap: int) -> None:
    date_text = item.get("date")
    if not date_text:
        raise ValueError("transcript_missing_date")
    date_year = datetime.strptime(date_text, "%Y-%m-%d").year
    if abs(date_year - requested_year) > max_year_gap:
        raise ValueError(
            f"date_year_mismatch: requested={requested_year}, transcript_date={date_text}"
        )


def compute_return_from_prices(
    prices: list[dict[str, Any]],
    return_window: int,
) -> dict[str, Any]:
    rows = sorted(prices, key=lambda row: row["date"])
    if len(rows) <= return_window:
        raise ValueError(f"not_enough_price_rows: rows={len(rows)}")

    price_day0 = float(rows[0]["close"])
    price_dayn = float(rows[return_window]["close"])
    return_pct = (price_dayn - price_day0) / price_day0 * 100
    return {
        "price_day0": price_day0,
        f"price_day{return_window}": price_dayn,
        "return_pct": return_pct,
        "return_window_trading_days": return_window,
    }


def build_record(
    client: FMPClient,
    symbol: str,
    year: int,
    quarter: int,
    return_window: int,
    price_cache: dict[str, Any],
    max_year_gap: int,
) -> dict[str, Any] | None:
    item = client.transcript(symbol, year, quarter)
    if item is None:
        return None

    validate_transcript_date(item, year, max_year_gap)

    period_quarter = quarter_from_period(item.get("period"))
    if period_quarter is not None and period_quarter != quarter:
        raise ValueError(
            f"quarter_mismatch: requested=Q{quarter}, transcript_period={item.get('period')}"
        )

    content = str(item.get("content") or "").strip()
    if not content:
        raise ValueError("empty_transcript_content")

    earnings_date = item["date"]
    cache_key = f"{symbol}_{earnings_date}_{return_window}"
    if cache_key in price_cache:
        price_result = price_cache[cache_key]
        if "error" in price_result:
            raise ValueError(price_result["error"])
    else:
        try:
            prices = client.prices(symbol, earnings_date)
            price_result = compute_return_from_prices(prices, return_window)
        except Exception as exc:
            price_result = {"error": str(exc)}
        price_cache[cache_key] = price_result
        if "error" in price_result:
            raise ValueError(price_result["error"])

    return {
        "symbol": symbol,
        "quarter": f"{year}Q{quarter}",
        "year": year,
        "period": f"Q{quarter}",
        "earnings_date": earnings_date,
        "source": "FMP",
        "transcript": content,
        **price_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("notebooks/data/fmp_model_ready"))
    parser.add_argument("--raw_dir", type=Path, default=Path("notebooks/data/fmp_transcripts_raw"))
    parser.add_argument("--ticker_file", type=Path, default=None)
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated ticker override.")
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--return_window", type=int, default=3)
    parser.add_argument("--max_records", type=int, default=3000)
    parser.add_argument("--max_year_gap", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise SystemExit("Set FMP_API_KEY in your environment or .env file.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        args.raw_dir.mkdir(parents=True, exist_ok=True)

    price_cache_path = output_dir / "_price_cache.json"
    manifest_path = output_dir / "_manifest.json"
    failures_path = output_dir / "_failures.json"
    price_cache = load_json(price_cache_path, {})
    manifest = []
    failures = []

    client = FMPClient(api_key=api_key, sleep_seconds=args.sleep)
    tickers = load_tickers(args)
    years = range(args.start_year, args.end_year + 1)
    quarters = [1, 2, 3, 4]

    written = len([p for p in output_dir.glob("*.json") if not p.name.startswith("_")])
    print(f"Tickers: {len(tickers)} | years: {args.start_year}-{args.end_year}")
    print(f"Existing model-ready records: {written}")

    for symbol in tickers:
        for year in years:
            for quarter in quarters:
                if written >= args.max_records:
                    save_json(price_cache_path, price_cache)
                    save_json(manifest_path, manifest)
                    save_json(failures_path, failures)
                    print(f"Reached max_records={args.max_records}")
                    print(f"Model-ready records: {written}")
                    return 0

                target = output_dir / f"{symbol}_{year}Q{quarter}.json"
                if target.exists() and not args.overwrite:
                    continue

                try:
                    record = build_record(
                        client,
                        symbol,
                        year,
                        quarter,
                        args.return_window,
                        price_cache,
                        args.max_year_gap,
                    )
                    if record is None:
                        continue

                    if args.save_raw:
                        raw_target = args.raw_dir / target.name
                        with raw_target.open("w") as f:
                            json.dump(
                                {
                                    "symbol": symbol,
                                    "year": year,
                                    "quarter": quarter,
                                    "source": "FMP",
                                    "transcript_raw": [
                                        {
                                            "symbol": symbol,
                                            "period": f"Q{quarter}",
                                            "year": year,
                                            "date": record["earnings_date"],
                                            "content": record["transcript"],
                                        }
                                    ],
                                },
                                f,
                                indent=2,
                            )

                    with target.open("w") as f:
                        json.dump(record, f, indent=2)
                    written += 1
                    manifest.append(
                        {
                            "json": str(target),
                            "symbol": symbol,
                            "quarter": record["quarter"],
                            "earnings_date": record["earnings_date"],
                            "return_pct": record["return_pct"],
                            "status": "written",
                        }
                    )

                    if written % 25 == 0:
                        save_json(price_cache_path, price_cache)
                        save_json(manifest_path, manifest)
                        save_json(failures_path, failures)
                        print(f"written={written} latest={target.name}")

                except Exception as exc:
                    failures.append(
                        {
                            "symbol": symbol,
                            "year": year,
                            "quarter": quarter,
                            "error": str(exc),
                        }
                    )

    save_json(price_cache_path, price_cache)
    save_json(manifest_path, manifest)
    save_json(failures_path, failures)
    print(f"Model-ready records: {written}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
