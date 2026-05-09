"""
Data Cleaner Module
Handles cleaning and preprocessing of stock market data
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCleaner:
    """
    Comprehensive data cleaning and preprocessing for stock market data
    """
    
    def __init__(self):
        """Initialize the data cleaner"""
        self.cleaning_report = {}
        
    def clean_data(self, df: pd.DataFrame, symbol: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main cleaning function that applies all cleaning steps
        
        Args:
            df: Raw stock data DataFrame
            symbol: Stock symbol for reporting
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        if df.empty:
            logger.error("Empty DataFrame provided")
            return df, {"error": "Empty DataFrame"}
        
        logger.info(f"Starting data cleaning for {symbol or 'unknown symbol'}")
        logger.info(f"Original data shape: {df.shape}")
        
        # Initialize report
        report = {
            'symbol': symbol,
            'original_shape': df.shape,
            'cleaning_steps': []
        }
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Apply cleaning steps
        df_clean, step_report = self._remove_duplicates(df_clean)
        report['cleaning_steps'].append(step_report)
        
        df_clean, step_report = self._handle_missing_values(df_clean)
        report['cleaning_steps'].append(step_report)
        
        df_clean, step_report = self._handle_outliers(df_clean)
        report['cleaning_steps'].append(step_report)
        
        df_clean, step_report = self._validate_price_relationships(df_clean)
        report['cleaning_steps'].append(step_report)
        
        df_clean, step_report = self._standardize_data_types(df_clean)
        report['cleaning_steps'].append(step_report)
        
        df_clean, step_report = self._sort_and_index(df_clean)
        report['cleaning_steps'].append(step_report)
        
        # Final report
        report['final_shape'] = df_clean.shape
        report['data_quality_score'] = self._calculate_quality_score(df_clean)
        report['cleaning_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Cleaning completed. Final shape: {df_clean.shape}")
        logger.info(f"Data quality score: {report['data_quality_score']:.2f}")
        
        return df_clean, report
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicate rows based on trading date and symbol
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, step report)
        """
        original_count = len(df)
        
        # Check for duplicates
        if 'trading_date' in df.columns and 'symbol' in df.columns:
            duplicates = df.duplicated(subset=['trading_date', 'symbol'], keep='first')
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate rows")
                df_clean = df[~duplicates].copy()
            else:
                df_clean = df.copy()
        else:
            logger.warning("Missing trading_date or symbol columns for duplicate detection")
            df_clean = df.copy()
            duplicate_count = 0
        
        report = {
            'step': 'remove_duplicates',
            'original_count': original_count,
            'duplicate_count': duplicate_count,
            'final_count': len(df_clean),
            'removed_count': original_count - len(df_clean)
        }
        
        return df_clean, report
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, step report)
        """
        original_count = len(df)
        df_clean = df.copy()
        
        # Check for missing values
        missing_info = {}
        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = missing_count
        
        report = {
            'step': 'handle_missing_values',
            'missing_info': missing_info,
            'strategies_used': []
        }
        
        # Handle missing values based on column type
        if missing_info:
            logger.info(f"Found missing values in columns: {list(missing_info.keys())}")
            
            # For price columns, use forward fill
            price_columns = ['open', 'high', 'low', 'close', 'adj_close']
            for col in price_columns:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    before_count = df_clean[col].isnull().sum()
                    df_clean[col] = df_clean[col].fillna(method='ffill')
                    after_count = df_clean[col].isnull().sum()
                    report['strategies_used'].append(f"Forward fill {col}: {before_count} -> {after_count}")
            
            # For volume, use forward fill then backward fill
            if 'volume' in df_clean.columns and df_clean['volume'].isnull().any():
                before_count = df_clean['volume'].isnull().sum()
                df_clean['volume'] = df_clean['volume'].fillna(method='ffill').fillna(method='bfill')
                after_count = df_clean['volume'].isnull().sum()
                report['strategies_used'].append(f"Forward/backward fill volume: {before_count} -> {after_count}")
            
            # For remaining missing values, drop rows
            remaining_missing = df_clean.isnull().any(axis=1).sum()
            if remaining_missing > 0:
                df_clean = df_clean.dropna()
                report['strategies_used'].append(f"Dropped {remaining_missing} rows with remaining missing values")
        
        report['final_count'] = len(df_clean)
        report['removed_count'] = original_count - len(df_clean)
        
        return df_clean, report
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and handle outliers in price and volume data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, step report)
        """
        df_clean = df.copy()
        report = {
            'step': 'handle_outliers',
            'outliers_detected': {},
            'outliers_handled': {}
        }
        
        # Check for outliers in price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df_clean.columns:
                # Calculate IQR
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Detect outliers
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    report['outliers_detected'][col] = outlier_count
                    
                    # Cap outliers instead of removing them
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    report['outliers_handled'][col] = f"Capped {outlier_count} outliers"
        
        # Check for negative or zero prices
        for col in price_columns:
            if col in df_clean.columns:
                negative_count = (df_clean[col] <= 0).sum()
                if negative_count > 0:
                    report['outliers_detected'][f'{col}_non_positive'] = negative_count
                    # Replace with forward fill or small positive value
                    df_clean.loc[df_clean[col] <= 0, col] = df_clean[col].median()
                    report['outliers_handled'][f'{col}_non_positive'] = f"Replaced {negative_count} non-positive values"
        
        # Check for negative volume
        if 'volume' in df_clean.columns:
            negative_volume = (df_clean['volume'] < 0).sum()
            if negative_volume > 0:
                report['outliers_detected']['negative_volume'] = negative_volume
                df_clean.loc[df_clean['volume'] < 0, 'volume'] = 0
                report['outliers_handled']['negative_volume'] = f"Set {negative_volume} negative volumes to 0"
        
        return df_clean, report
    
    def _validate_price_relationships(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and correct price relationships (high >= low, etc.)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, step report)
        """
        df_clean = df.copy()
        report = {
            'step': 'validate_price_relationships',
            'corrections': {}
        }
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df_clean.columns for col in required_columns):
            report['error'] = "Missing required price columns"
            return df_clean, report
        
        # Ensure high >= low
        invalid_high_low = df_clean['high'] < df_clean['low']
        if invalid_high_low.any():
            count = invalid_high_low.sum()
            report['corrections']['high_low_swap'] = f"Swapped high/low for {count} rows"
            
            # Swap high and low where needed
            df_clean.loc[invalid_high_low, ['high', 'low']] = df_clean.loc[invalid_high_low, ['low', 'high']].values
        
        # Ensure high >= open and close
        high_less_open = df_clean['high'] < df_clean['open']
        if high_less_open.any():
            count = high_less_open.sum()
            report['corrections']['high_open'] = f"Adjusted high for {count} rows where high < open"
            df_clean.loc[high_less_open, 'high'] = df_clean.loc[high_less_open, ['open', 'close']].max(axis=1)
        
        high_less_close = df_clean['high'] < df_clean['close']
        if high_less_close.any():
            count = high_less_close.sum()
            report['corrections']['high_close'] = f"Adjusted high for {count} rows where high < close"
            df_clean.loc[high_less_close, 'high'] = df_clean.loc[high_less_close, ['open', 'close']].max(axis=1)
        
        # Ensure low <= open and close
        low_greater_open = df_clean['low'] > df_clean['open']
        if low_greater_open.any():
            count = low_greater_open.sum()
            report['corrections']['low_open'] = f"Adjusted low for {count} rows where low > open"
            df_clean.loc[low_greater_open, 'low'] = df_clean.loc[low_greater_open, ['open', 'close']].min(axis=1)
        
        low_greater_close = df_clean['low'] > df_clean['close']
        if low_greater_close.any():
            count = low_greater_close.sum()
            report['corrections']['low_close'] = f"Adjusted low for {count} rows where low > close"
            df_clean.loc[low_greater_close, 'low'] = df_clean.loc[low_greater_open, ['open', 'close']].min(axis=1)
        
        return df_clean, report
    
    def _standardize_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Standardize data types for consistency
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, step report)
        """
        df_clean = df.copy()
        report = {
            'step': 'standardize_data_types',
            'type_conversions': {}
        }
        
        # Convert date columns
        date_columns = ['trading_date', 'date']
        for col in date_columns:
            if col in df_clean.columns:
                if df_clean[col].dtype != 'datetime64[ns]':
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    report['type_conversions'][col] = f"Converted to datetime"
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df_clean.columns:
                if df_clean[col].dtype not in ['float64', 'int64']:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    report['type_conversions'][col] = f"Converted to numeric"
        
        # Convert symbol to string
        if 'symbol' in df_clean.columns:
            if df_clean['symbol'].dtype != 'object':
                df_clean['symbol'] = df_clean['symbol'].astype(str)
                report['type_conversions']['symbol'] = "Converted to string"
        
        return df_clean, report
    
    def _sort_and_index(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Sort data by date and set appropriate index
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, step report)
        """
        df_clean = df.copy()
        report = {
            'step': 'sort_and_index',
            'operations': []
        }
        
        # Sort by date
        date_columns = ['trading_date', 'date']
        date_col = None
        for col in date_columns:
            if col in df_clean.columns:
                date_col = col
                break
        
        if date_col:
            df_clean = df_clean.sort_values(date_col)
            report['operations'].append(f"Sorted by {date_col}")
            
            # Set date as index for time series operations
            if len(df_clean) > 0:
                df_clean = df_clean.set_index(date_col)
                report['operations'].append(f"Set {date_col} as index")
        
        return df_clean, report
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100)
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Quality score
        """
        if df.empty:
            return 0.0
        
        score = 100.0
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 30  # Deduct up to 30 points for missing data
        
        # Check for duplicates
        if len(df) > 0:
            duplicate_ratio = df.duplicated().sum() / len(df)
            score -= duplicate_ratio * 20  # Deduct up to 20 points for duplicates
        
        # Check data consistency
        required_columns = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in required_columns):
            # Check price relationships
            price_violations = 0
            price_violations += (df['high'] < df['low']).sum()
            price_violations += (df['high'] < df['open']).sum()
            price_violations += (df['high'] < df['close']).sum()
            price_violations += (df['low'] > df['open']).sum()
            price_violations += (df['low'] > df['close']).sum()
            
            if len(df) > 0:
                violation_ratio = price_violations / len(df)
                score -= violation_ratio * 50  # Deduct up to 50 points for price violations
        
        return max(0.0, min(100.0, score))
    
    def generate_cleaning_summary(self, report: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the cleaning process
        
        Args:
            report: Cleaning report dictionary
            
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append(f"Data Cleaning Summary for {report.get('symbol', 'Unknown Symbol')}")
        summary.append("=" * 50)
        summary.append(f"Original shape: {report['original_shape']}")
        summary.append(f"Final shape: {report['final_shape']}")
        summary.append(f"Data quality score: {report['data_quality_score']:.2f}/100")
        summary.append("")
        
        for step in report['cleaning_steps']:
            step_name = step.get('step', 'Unknown step')
            summary.append(f"Step: {step_name}")
            
            if 'duplicate_count' in step:
                summary.append(f"  - Duplicates removed: {step['duplicate_count']}")
            
            if 'missing_info' in step and step['missing_info']:
                summary.append(f"  - Missing values found: {step['missing_info']}")
            
            if 'strategies_used' in step:
                for strategy in step['strategies_used']:
                    summary.append(f"  - {strategy}")
            
            if 'outliers_detected' in step and step['outliers_detected']:
                summary.append(f"  - Outliers detected: {step['outliers_detected']}")
            
            if 'corrections' in step and step['corrections']:
                for correction in step['corrections'].values():
                    summary.append(f"  - {correction}")
            
            if 'type_conversions' in step and step['type_conversions']:
                for conversion in step['type_conversions'].values():
                    summary.append(f"  - {conversion}")
            
            summary.append("")
        
        return "\n".join(summary)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import StockDataFetcher
    
    print("Testing data cleaner...")
    
    # Fetch some data
    fetcher = StockDataFetcher()
    raw_data = fetcher.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    
    if not raw_data.empty:
        # Clean the data
        cleaner = StockDataCleaner()
        clean_data, report = cleaner.clean_data(raw_data, 'AAPL')
        
        print("\nCleaning Summary:")
        print(cleaner.generate_cleaning_summary(report))
        
        print(f"\nOriginal data shape: {raw_data.shape}")
        print(f"Cleaned data shape: {clean_data.shape}")
        print(f"Data quality score: {report['data_quality_score']:.2f}")
    else:
        print("No data available for testing")
