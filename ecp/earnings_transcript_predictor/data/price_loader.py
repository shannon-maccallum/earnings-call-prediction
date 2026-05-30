"""
price_loader.py

Calculates post-earnings stock returns using Yahoo Finance price data.

Per instructor feedback:
  - Return window is configurable (1, 3, or 5 trading days).
  - We use the closing price on the earnings date as the baseline.
    If the call happened after market close, the "day 0" close already
    reflects after-hours reaction; the return then captures the subsequent
    drift. This is a known limitation noted in the milestone report.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class PriceLoader:
    """
    Fetches stock price data and computes post-earnings returns.

    Args:
        return_window (int): Number of trading days after earnings to
                             measure the return. Default is 3.
                             - 1 day:  short-term reaction
                             - 3 days: standard event-study window
                             - 5 days: one full trading week
    """

    def __init__(self, return_window: int = 3):
        if return_window not in (1, 3, 5):
            raise ValueError("return_window must be 1, 3, or 5 trading days.")
        self.return_window = return_window

    def get_return(self, symbol: str, earnings_date: str) -> dict:
        """
        Compute the post-earnings return for a single stock.

        Args:
            symbol:        Ticker symbol, e.g. "AAPL"
            earnings_date: Date of earnings call, "YYYY-MM-DD"

        Returns:
            dict with keys:
                - symbol (str)
                - earnings_date (str)
                - return_window (int): days used
                - price_day0 (float): closing price on earnings date
                - price_dayN (float): closing price N trading days later
                - return_pct (float): percentage return (target variable)
                - direction (int): 1 if positive return, 0 if negative
        """
        date0 = pd.Timestamp(earnings_date)
        # Fetch enough data to cover the window + buffer for weekends/holidays
        fetch_end = date0 + timedelta(days=self.return_window * 2 + 5)

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=earnings_date, end=fetch_end.strftime("%Y-%m-%d"))

        if hist.empty or len(hist) < 2:
            raise ValueError(
                f"Not enough price data for {symbol} around {earnings_date}. "
                "The earnings date may be in the future or a holiday."
            )

        # Day 0: first available close on or after earnings_date
        price_day0 = float(hist["Close"].iloc[0])

        # Day N: close after return_window trading days
        if len(hist) <= self.return_window:
            raise ValueError(
                f"Only {len(hist)} trading days of data for {symbol} "
                f"but need {self.return_window + 1}."
            )
        price_dayN = float(hist["Close"].iloc[self.return_window])

        return_pct = (price_dayN - price_day0) / price_day0 * 100

        return {
            "symbol": symbol,
            "earnings_date": earnings_date,
            "return_window": self.return_window,
            "price_day0": round(price_day0, 4),
            "price_dayN": round(price_dayN, 4),
            "return_pct": round(return_pct, 4),
            "direction": 1 if return_pct >= 0 else 0,
        }

    def get_returns_batch(self, requests_list: list[tuple[str, str]]) -> pd.DataFrame:
        """
        Compute returns for a list of (symbol, earnings_date) pairs.

        Args:
            requests_list: e.g. [("AAPL", "2024-02-01"), ("MSFT", "2024-01-24")]

        Returns:
            DataFrame with one row per earnings call and columns:
            symbol, earnings_date, return_window, price_day0,
            price_dayN, return_pct, direction
        """
        rows = []
        for symbol, date in requests_list:
            try:
                row = self.get_return(symbol, date)
                rows.append(row)
            except Exception as e:
                print(f"  Warning: skipping {symbol} {date}: {e}")
        return pd.DataFrame(rows)
