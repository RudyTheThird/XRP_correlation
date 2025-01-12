"""
Cryptocurrency Correlation Analysis Script

This script fetches historical price data for XRP and Coinbase-listed assets,
calculates rolling correlations and beta values, and exports analysis results.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, TypedDict, Union

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt

class CryptoData(TypedDict):
    """Type definition for cryptocurrency price data."""
    symbol: str
    prices: List[float]
    timestamps: List[str]

class CorrelationResult(TypedDict):
    """Type definition for correlation analysis results."""
    symbol: str
    correlation: float
    beta: float
    correlation_stability: float

class CryptoAnalyzer:
    """Handles cryptocurrency data fetching and analysis."""
    
    def __init__(self, api_key: str, base_url: str = "https://rest.coinapi.io/v1"):
        """
        Initialize the analyzer with API credentials.
        
        Args:
            api_key: CoinAPI authentication key
            base_url: Base URL for CoinAPI endpoints
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-CoinAPI-Key": api_key}
        self.max_retries = 5
        self.base_delay = 1  # Base delay for exponential backoff in seconds
        self._symbols_cache: Optional[List[Dict]] = None

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make an API request with exponential backoff retry logic.
        
        Args:
            endpoint: API endpoint to query
            params: Optional query parameters
            
        Returns:
            API response data
            
        Raises:
            requests.exceptions.RequestException: If request fails after max retries
        """
        url = f"{self.base_url}/{endpoint}"
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params,
                    timeout=30  # Add timeout to prevent hanging
                )
                
                if response.status_code == 429:  # Rate limit exceeded
                    retry_count += 1
                    delay = (2 ** retry_count * self.base_delay + 
                            np.random.uniform(0, 0.1 * self.base_delay))
                    time.sleep(delay)
                    continue
                    
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise e
                    
                # Exponential backoff with jitter
                delay = (2 ** retry_count * self.base_delay + 
                        np.random.uniform(0, 0.1 * self.base_delay))
                time.sleep(delay)
        
        raise RuntimeError("Request failed after max retries")

    def _get_symbols(self) -> List[Dict]:
        """
        Get and cache exchange symbols.
        
        Returns:
            List of symbol dictionaries
        """
        if self._symbols_cache is None:
            print("Fetching and caching exchange symbols...")
            self._symbols_cache = self._make_request("symbols")
        return self._symbols_cache

    def get_coinbase_assets(self) -> List[str]:
        """
        Fetch list of assets traded on Coinbase.
        
        Returns:
            List of asset symbols
        """
        try:
            symbols = self._get_symbols()
            
            # Filter for Coinbase SPOT trading pairs with USD
            coinbase_assets = set()
            for symbol in symbols:
                if (symbol["exchange_id"] == "COINBASE" and 
                    symbol["symbol_type"] == "SPOT" and 
                    symbol["asset_id_quote"] == "USD"):
                    coinbase_assets.add(symbol["asset_id_base"])
            
            return list(coinbase_assets)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Coinbase assets: {str(e)}")

    def get_historical_data(
        self, 
        symbol: str, 
        days: int = 90,
        period: str = "1HRS"
    ) -> CryptoData:
        """
        Fetch historical price data for a given symbol.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of historical data
            period: Time period for data points
            
        Returns:
            Dictionary containing symbol, prices, and timestamps
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Format timestamps according to ISO 8601
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        # Get symbol ID for the specific exchange pair
        try:
            symbols = self._get_symbols()
            symbol_id = next(
                (s["symbol_id"] for s in symbols 
                 if s["exchange_id"] == "COINBASE" and 
                 s["asset_id_base"] == symbol and 
                 s["asset_id_quote"] == "USD" and
                 s["symbol_type"] == "SPOT"),
                None
            )
            
            if not symbol_id:
                raise ValueError(f"Symbol {symbol}/USD not found on Coinbase")
            
            endpoint = f"ohlcv/{symbol_id}/history"
            params = {
                "period_id": period,
                "time_start": start_str,
                "time_end": end_str,
                "limit": 2000  # Maximum allowed by API
            }
            
            data = self._make_request(endpoint, params)
            if not data:
                raise ValueError(f"No data returned for {symbol}")
                
            return {
                "symbol": symbol,
                "prices": [float(rate["price_close"]) for rate in data],
                "timestamps": [rate["time_period_end"] for rate in data]
            }
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch historical data for {symbol}: {str(e)}"
            )

    def calculate_metrics(
        self,
        base_data: CryptoData,
        compare_data: CryptoData,
        window: int = 168  # 7 days in hours
    ) -> CorrelationResult:
        """
        Calculate correlation and beta metrics between two assets.
        
        Args:
            base_data: Price data for base asset (XRP)
            compare_data: Price data for comparison asset
            window: Rolling window size in hours
            
        Returns:
            Dictionary containing correlation and beta metrics
        """
        try:
            # Create DataFrames with aligned timestamps
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(base_data["timestamps"]),
                "base_price": base_data["prices"]
            }).set_index("timestamp")
            
            compare_df = pd.DataFrame({
                "timestamp": pd.to_datetime(compare_data["timestamps"]),
                "compare_price": compare_data["prices"]
            }).set_index("timestamp")
            
            # Align the data on timestamps
            df = df.join(compare_df, how="inner")
            
            if len(df) < window:
                raise ValueError(
                    f"Insufficient data points ({len(df)}) for window size {window}"
                )
            
            # Calculate returns
            df["base_returns"] = df["base_price"].pct_change()
            df["compare_returns"] = df["compare_price"].pct_change()
            
            # Remove any NaN values
            df = df.dropna()
            
            if len(df) < window:
                raise ValueError(
                    "Insufficient valid data points after cleaning"
                )
            
            # Calculate rolling correlation
            rolling_corr = df["base_returns"].rolling(window=window).corr(
                df["compare_returns"]
            )
            
            # Calculate rolling beta
            rolling_cov = df["base_returns"].rolling(window=window).cov(
                df["compare_returns"]
            )
            rolling_var = df["compare_returns"].rolling(window=window).var()
            rolling_beta = rolling_cov / rolling_var
            
            # Calculate correlation stability (standard deviation of rolling correlation)
            correlation_stability = rolling_corr.std()
            
            return {
                "symbol": compare_data["symbol"],
                "correlation": float(rolling_corr.iloc[-1]),
                "beta": float(rolling_beta.iloc[-1]),
                "correlation_stability": float(correlation_stability)
            }
        except Exception as e:
            raise ValueError(
                f"Failed to calculate metrics for {compare_data['symbol']}: {str(e)}"
            )

    def plot_correlation_stability(
        self,
        base_data: CryptoData,
        compare_data: CryptoData,
        window: int = 168,
        output_path: str = "correlation_stability.png"
    ) -> None:
        """
        Plot correlation stability over time.
        
        Args:
            base_data: Price data for base asset (XRP)
            compare_data: Price data for comparison asset
            window: Rolling window size in hours
            output_path: Path to save the plot
        """
        try:
            # Create DataFrames with aligned timestamps
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(base_data["timestamps"]),
                "base_price": base_data["prices"]
            }).set_index("timestamp")
            
            compare_df = pd.DataFrame({
                "timestamp": pd.to_datetime(compare_data["timestamps"]),
                "compare_price": compare_data["prices"]
            }).set_index("timestamp")
            
            # Align the data on timestamps
            df = df.join(compare_df, how="inner")
            
            # Calculate returns
            df["base_returns"] = df["base_price"].pct_change()
            df["compare_returns"] = df["compare_price"].pct_change()
            df = df.dropna()
            
            # Calculate rolling correlation
            rolling_corr = df["base_returns"].rolling(window=window).corr(
                df["compare_returns"]
            )
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(rolling_corr.index, rolling_corr.values)
            plt.title(f"Rolling Correlation: XRP vs {compare_data['symbol']}")
            plt.xlabel("Date")
            plt.ylabel("Correlation Coefficient")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            print(f"Failed to create plot for {compare_data['symbol']}: {str(e)}")

def main() -> None:
    """Main execution function."""
    API_KEY = "efb2594c-3c4d-4ed5-8266-bd11217f198b"
    analyzer = CryptoAnalyzer(API_KEY)
    
    try:
        # Get Coinbase assets first to validate XRP availability
        print("Fetching Coinbase assets...")
        assets = analyzer.get_coinbase_assets()
        
        if "XRP" not in assets:
            raise ValueError("XRP not available on Coinbase")
        
        # Fetch XRP data
        print("Fetching XRP historical data...")
        xrp_data = analyzer.get_historical_data("XRP")
        
        results: List[CorrelationResult] = []
        for asset in assets:
            if asset == "XRP":
                continue
                
            try:
                print(f"Analyzing {asset}...")
                asset_data = analyzer.get_historical_data(asset)
                result = analyzer.calculate_metrics(xrp_data, asset_data)
                results.append(result)
                
                # Generate correlation stability plot for this asset
                analyzer.plot_correlation_stability(
                    xrp_data,
                    asset_data,
                    output_path=f"correlation_stability_{asset}.png"
                )
                
            except Exception as e:
                print(f"Error analyzing {asset}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No valid correlation results were calculated")
        
        # Sort results by correlation (absolute value) and beta
        correlation_sorted = sorted(
            results,
            key=lambda x: abs(x["correlation"]),
            reverse=True
        )[:5]
        
        beta_sorted = sorted(
            results,
            key=lambda x: abs(x["beta"]),
            reverse=True
        )[:5]
        
        # Export results
        output = {
            "top_correlations": correlation_sorted,
            "top_betas": beta_sorted,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        with open("crypto_analysis_results.json", "w") as f:
            json.dump(output, f, indent=2)
            
        print("Analysis complete! Results saved to crypto_analysis_results.json")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()