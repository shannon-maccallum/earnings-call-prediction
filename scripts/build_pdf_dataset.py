#!/usr/bin/env python3
"""Convert cleaned earnings-call PDFs into model-ready JSON records.

The model trains on JSON records with a transcript string and a return_pct label.
This script extracts transcript text from cleaned S&P Global PDF transcripts,
derives ticker/quarter/call date from the normalized filename/header, and labels
each example with the post-call stock return from Yahoo Finance.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import yfinance as yf
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  python -m pip install pypdf yfinance pandas\n"
        f"\nOriginal error: {exc}"
    )

logging.getLogger("pypdf").setLevel(logging.ERROR)

FILENAME_RE = re.compile(
    r"^(?P<folder_ticker>[A-Z0-9]+)_(?P<year>\d{4})Q(?P<quarter>[1-4])_"
    r"(?P<date>\d{4}-\d{2}-\d{2})(?:_dup\d+)?\.pdf$"
)

HEADER_RE = re.compile(
    r"\bF?\s*Q\s*(?P<quarter>[1-4])\s+(?P<year>(?:19|20)\d{2})\s+"
    r"(?:[A-Za-z]+\s+){0,4}Earnings\s+Call",
    re.IGNORECASE,
)

YFINANCE_TICKERS = {
    "NVIDIA": "NVDA",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pdf_text(path: Path, max_pages: int | None = None) -> str:
    reader = PdfReader(str(path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    chunks = []
    for page in pages:
        chunks.append(page.extract_text() or "")
    return normalize_text("\n".join(chunks))


def parse_pdf_metadata(path: Path) -> dict[str, str]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"filename_not_normalized: {path.name}")

    page1 = extract_pdf_text(path, max_pages=1)
    header = HEADER_RE.search(page1[:2500])
    if not header:
        raise ValueError("quarter_header_not_found_on_page1")

    file_quarter = match.group("quarter")
    file_year = match.group("year")
    header_quarter = header.group("quarter")
    header_year = header.group("year")
    if file_quarter != header_quarter or file_year != header_year:
        raise ValueError(
            "filename_header_mismatch: "
            f"file={file_year}Q{file_quarter}, header={header_year}Q{header_quarter}"
        )

    folder_ticker = path.parent.name
    return {
        "folder_ticker": folder_ticker,
        "symbol": YFINANCE_TICKERS.get(folder_ticker, folder_ticker),
        "quarter": f"{file_year}Q{file_quarter}",
        "fiscal_year": file_year,
        "fiscal_quarter": f"Q{file_quarter}",
        "earnings_date": match.group("date"),
    }


def load_price_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_price_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def compute_return(
    symbol: str,
    earnings_date: str,
    return_window: int,
    price_cache: dict[str, Any],
) -> dict[str, Any]:
    cache_key = f"{symbol}_{earnings_date}_{return_window}"
    if cache_key in price_cache:
        cached = dict(price_cache[cache_key])
        if "error" in cached:
            raise ValueError(cached["error"])
        return cached

    start = datetime.strptime(earnings_date, "%Y-%m-%d")
    end = start + timedelta(days=return_window + 14)
    prices = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if len(prices) <= return_window:
        error = f"not_enough_price_rows: {symbol} {earnings_date} rows={len(prices)}"
        price_cache[cache_key] = {"error": error}
        raise ValueError(error)

    close = prices["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    price_day0 = float(close.iloc[0])
    price_dayn = float(close.iloc[return_window])
    return_pct = (price_dayn - price_day0) / price_day0 * 100
    result = {
        "price_day0": price_day0,
        f"price_day{return_window}": price_dayn,
        "return_pct": return_pct,
        "return_window_trading_days": return_window,
    }
    price_cache[cache_key] = result
    return result


def output_name(meta: dict[str, str], source_name: str) -> str:
    stem = Path(source_name).stem
    dup = ""
    dup_match = re.search(r"(_dup\d+)$", stem)
    if dup_match:
        dup = dup_match.group(1)
    return f"{meta['symbol']}_{meta['quarter']}{dup}.json"


def build_record(
    pdf_path: Path,
    return_window: int,
    price_cache: dict[str, Any],
) -> dict[str, Any]:
    meta = parse_pdf_metadata(pdf_path)
    transcript = extract_pdf_text(pdf_path)
    prices = compute_return(meta["symbol"], meta["earnings_date"], return_window, price_cache)
    return {
        **meta,
        **prices,
        "transcript": transcript,
        "source_pdf": str(pdf_path),
        "source": "cleaned_pdf_archive",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf_root",
        type=Path,
        default=Path("~/Library/CloudStorage/OneDrive-UniversityOfOregon/transcripts").expanduser(),
        help="Root folder containing ticker subfolders of cleaned PDFs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("notebooks/data/pdf_transcripts"),
        help="Directory where model-ready JSON records will be written.",
    )
    parser.add_argument("--return_window", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for testing.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds between Yahoo calls.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pdf_root = args.pdf_root.expanduser().resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "_manifest.json"
    failures_path = output_dir / "_failures.json"
    price_cache_path = output_dir / "_price_cache.json"

    pdfs = sorted(
        p
        for p in pdf_root.glob("*/*.pdf")
        if p.parent.name not in {"excluded", "needs_review"}
    )
    if args.limit:
        pdfs = pdfs[: args.limit]

    price_cache = load_price_cache(price_cache_path)
    manifest = []
    failures = []

    for index, pdf_path in enumerate(pdfs, start=1):
        try:
            meta = parse_pdf_metadata(pdf_path)
            target = output_dir / output_name(meta, pdf_path.name)
            if target.exists() and not args.overwrite:
                manifest.append(
                    {
                        "pdf": str(pdf_path),
                        "json": str(target),
                        "status": "exists",
                        "symbol": meta["symbol"],
                        "quarter": meta["quarter"],
                    }
                )
                continue

            record = build_record(pdf_path, args.return_window, price_cache)
            with target.open("w") as f:
                json.dump(record, f, indent=2)
            manifest.append(
                {
                    "pdf": str(pdf_path),
                    "json": str(target),
                    "status": "written",
                    "symbol": record["symbol"],
                    "quarter": record["quarter"],
                    "earnings_date": record["earnings_date"],
                    "return_pct": record["return_pct"],
                }
            )
            time.sleep(args.sleep)
        except Exception as exc:
            failures.append({"pdf": str(pdf_path), "error": str(exc)})

        if index % 25 == 0:
            save_price_cache(price_cache_path, price_cache)
            with manifest_path.open("w") as f:
                json.dump(manifest, f, indent=2)
            with failures_path.open("w") as f:
                json.dump(failures, f, indent=2)
            print(f"Processed {index}/{len(pdfs)} | ok={len(manifest)} failed={len(failures)}")

    save_price_cache(price_cache_path, price_cache)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    with failures_path.open("w") as f:
        json.dump(failures, f, indent=2)

    print(f"PDFs scanned: {len(pdfs)}")
    record_count = len([p for p in output_dir.glob("*.json") if not p.name.startswith("_")])
    print(f"JSON records ready: {record_count}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Failures file: {failures_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
