"""
Data Fetcher Module
Handles fetching stock market data from various sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import logging
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """
    A comprehensive stock data fetcher supporting multiple data sources
    """
    
    def __init__(self):
        """Initialize the data fetcher with default settings"""
        self.cache = {}
        self.rate_limit_delay = 1  # seconds between requests
        self.max_retries = 3
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol and date range
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.cache:
            logger.info(f"Returning cached data for {symbol}")
            return self.cache[cache_key]
        
        # Fetch from Yahoo Finance
        df = self._fetch_from_yahoo(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            # Cache the result
            self.cache[cache_key] = df
            logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
            return df
        else:
            logger.error(f"Failed to fetch data for {symbol}")
            return pd.DataFrame()
    
    def _fetch_from_yahoo(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance with retry logic
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with stock data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching {symbol} data from Yahoo Finance (attempt {attempt + 1})")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Download historical data
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                
                # Standardize column names
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                
                # Reset index to make date a column
                df.reset_index(inplace=True)
                df.rename(columns={'date': 'trading_date'}, inplace=True)
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Validate data
                if self._validate_data(df):
                    return df
                else:
                    logger.warning(f"Data validation failed for {symbol}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                else:
                    logger.error(f"Max retries exceeded for {symbol}")
                    return None
        
        return None
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the fetched data for quality and completeness
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['trading_date', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            logger.error("Missing required columns")
            return False
        
        # Check for empty data
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        # Check for negative prices or volume
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                logger.error(f"Found non-positive values in {col}")
                return False
        
        if (df['volume'] < 0).any():
            logger.error("Found negative volume values")
            return False
        
        # Check for reasonable price relationships
        if not (df['high'] >= df['low']).all():
            logger.error("High prices are not always >= low prices")
            return False
        
        if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            logger.error("High prices are not always >= open and close")
            return False
        
        if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
            logger.error("Low prices are not always <= open and close")
            return False
        
        return True
    
    def fetch_multiple_stocks(self, symbols: List[str], start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.fetch_data(symbol, start_date, end_date)
            if not df.empty:
                results[symbol] = df
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        return results
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get basic stock information
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            return False
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filename)
            logger.info(f"Data loaded from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading from CSV: {str(e)}")
            return pd.DataFrame()

# Example usage and testing
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    # Test single stock fetch
    print("Testing single stock fetch...")
    apple_data = fetcher.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    if not apple_data.empty:
        print(f"Successfully fetched {len(apple_data)} days of AAPL data")
        print(apple_data.head())
        print("\nData columns:", apple_data.columns.tolist())
    
    # Test stock info
    print("\nTesting stock info fetch...")
    apple_info = fetcher.get_stock_info('AAPL')
    print("Apple Stock Info:")
    for key, value in apple_info.items():
        print(f"  {key}: {value}")
    
    # Test multiple stocks
    print("\nTesting multiple stocks fetch...")
    symbols = ['MSFT', 'GOOGL', 'TSLA']
    multi_data = fetcher.fetch_multiple_stocks(symbols, '2023-01-01', '2023-12-31')
    print(f"Successfully fetched data for {len(multi_data)} stocks")
    
    # Test CSV save/load
    if not apple_data.empty:
        print("\nTesting CSV save/load...")
        fetcher.save_to_csv(apple_data, '../data/apple_test.csv')
        loaded_data = fetcher.load_from_csv('../data/apple_test.csv')
        print(f"Loaded {len(loaded_data)} rows from CSV")
