"""
transcript_loader.py

Fetches earnings call transcripts from the Alpha Vantage API and caches
them locally as JSON so we don't burn API calls on every run.
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class TranscriptLoader:
    """
    Loads earnings call transcripts from Alpha Vantage.

    Alpha Vantage endpoint:
        GET https://www.alphavantage.co/query
            ?function=EARNINGS_CALL_TRANSCRIPT
            &symbol=AAPL
            &quarter=2024Q1
            &apikey=YOUR_KEY

    Per instructor feedback: transcripts are cached locally after the first
    fetch so we never hit the API twice for the same transcript.

    Args:
        api_key (str): Alpha Vantage API key. Defaults to ALPHA_VANTAGE_API_KEY env var.
        cache_dir (str): Directory to store cached JSON transcripts.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = None, cache_dir: str = "data/transcripts"):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set ALPHA_VANTAGE_API_KEY in your .env file "
                "or pass api_key= to TranscriptLoader()."
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, quarter: str) -> Path:
        return self.cache_dir / f"{symbol}_{quarter}.json"

    def _fetch_from_api(self, symbol: str, quarter: str) -> dict:
        """Make a live API call to Alpha Vantage."""
        params = {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": symbol,
            "quarter": quarter,
            "apikey": self.api_key,
        }
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Alpha Vantage returns an error note in the JSON body on failures
        if "Note" in data:
            raise RuntimeError(
                f"Alpha Vantage rate limit hit: {data['Note']}. "
                "Free tier allows 25 requests/day."
            )
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error for {symbol} {quarter}: {data['Error Message']}")

        return data

    def load(self, symbol: str, quarter: str, force_refresh: bool = False) -> dict:
        """
        Load a transcript for a given ticker and quarter.

        Checks the local cache first. If not cached (or force_refresh=True),
        fetches from the API and saves to disk.

        Args:
            symbol:  Ticker symbol, e.g. "AAPL"
            quarter: Quarter string, e.g. "2024Q1"
            force_refresh: If True, bypass cache and re-fetch.

        Returns:
            dict with keys:
                - symbol (str)
                - quarter (str)
                - transcript (str): full transcript text
                - fiscalDateEnding (str): date string "YYYY-MM-DD"
        """
        cache_file = self._cache_path(symbol, quarter)

        if cache_file.exists() and not force_refresh:
            with open(cache_file, "r") as f:
                return json.load(f)

        print(f"Fetching {symbol} {quarter} from Alpha Vantage API...")
        data = self._fetch_from_api(symbol, quarter)

        # Cache to disk immediately
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Cached to {cache_file}")

        return data

    def load_batch(
        self,
        requests_list: list[tuple[str, str]],
        delay_seconds: float = 12.0,
    ) -> list[dict]:
        """
        Load multiple transcripts, respecting API rate limits.

        Alpha Vantage free tier: 5 calls/minute, 25/day.
        Default delay of 12s between calls keeps us under 5/min.
        Cached transcripts skip the delay entirely.

        Args:
            requests_list: List of (symbol, quarter) tuples.
                           e.g. [("AAPL", "2024Q1"), ("MSFT", "2024Q1")]
            delay_seconds: Seconds to wait between live API calls.

        Returns:
            List of transcript dicts (same format as load()).
        """
        results = []
        for i, (symbol, quarter) in enumerate(requests_list):
            cache_file = self._cache_path(symbol, quarter)
            is_cached = cache_file.exists()

            result = self.load(symbol, quarter)
            results.append(result)

            # Only sleep if we actually hit the API
            if not is_cached and i < len(requests_list) - 1:
                print(f"  Waiting {delay_seconds}s to respect rate limit...")
                time.sleep(delay_seconds)

        return results

    def get_transcript_text(self, transcript_data: dict) -> str:
        """Extract just the transcript text string from the API response dict."""
        # Alpha Vantage returns transcript under the "transcript" key
        return transcript_data.get("transcript", "")
