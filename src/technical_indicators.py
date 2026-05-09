"""
Technical Indicators Module
Calculates various technical indicators for stock market analysis
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator"""
        self.indicators_cache = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            logger.error("Empty DataFrame provided")
            return df
        
        logger.info("Calculating all technical indicators...")
        df_indicators = df.copy()
        
        # Calculate different categories of indicators
        df_indicators = self._calculate_moving_averages(df_indicators)
        df_indicators = self._calculate_momentum_indicators(df_indicators)
        df_indicators = self._calculate_volatility_indicators(df_indicators)
        df_indicators = self._calculate_volume_indicators(df_indicators)
        df_indicators = self._calculate_trend_indicators(df_indicators)
        
        logger.info(f"Calculated {len(df_indicators.columns) - len(df.columns)} new indicators")
        return df_indicators
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various moving averages
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with moving averages added
        """
        if 'close' not in df.columns:
            logger.error("Close price column not found")
            return df
        
        # Simple Moving Averages
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Weighted Moving Average
        for period in [10, 20, 50]:
            weights = np.arange(1, period + 1)
            df[f'wma_{period}'] = df['close'].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with momentum indicators added
        """
        if 'close' not in df.columns:
            logger.error("Close price column not found")
            return df
        
        try:
            # RSI (Relative Strength Index)
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_30'] = ta.momentum.RSIIndicator(df['close'], window=30).rsi()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Rate of Change (ROC)
            df['roc_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
            df['roc_20'] = ta.momentum.ROCIndicator(df['close'], window=20).roc()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Money Flow Index (MFI)
            if 'volume' in df.columns:
                df['mfi_14'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volatility indicators added
        """
        if 'close' not in df.columns:
            logger.error("Close price column not found")
            return df
        
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Average True Range (ATR)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # Historical Volatility
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['volatility_50'] = df['returns'].rolling(window=50).std() * np.sqrt(252)
            
            # Keltner Channels
            if all(col in df.columns for col in ['high', 'low', 'close']):
                kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
                df['kc_upper'] = kc.keltner_channel_hband()
                df['kc_middle'] = kc.keltner_channel_mband()
                df['kc_lower'] = kc.keltner_channel_lband()
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volume indicators added
        """
        if 'volume' not in df.columns:
            logger.warning("Volume column not found, skipping volume indicators")
            return df
        
        try:
            # Volume Moving Average
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_sma_50'] = df['volume'].rolling(window=50).mean()
            
            # On-Balance Volume (OBV)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # Volume Price Trend (VPT)
            df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
            
            # Accumulation/Distribution Line
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['adl'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
            
            # Chaikin Money Flow (CMF)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
            
            # Ease of Movement
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['emv_14'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).ease_of_movement()
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
        
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with trend indicators added
        """
        if 'close' not in df.columns:
            logger.error("Close price column not found")
            return df
        
        try:
            # ADX (Average Directional Index)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
                df['adx'] = adx.adx()
                df['adx_pos'] = adx.adx_pos()
                df['adx_neg'] = adx.adx_neg()
            
            # Aroon Indicator
            if all(col in df.columns for col in ['high', 'low']):
                aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
                df['aroon_up'] = aroon.aroon_up()
                df['aroon_down'] = aroon.aroon_down()
                df['aroon_indicator'] = aroon.aroon_indicator()
            
            # Commodity Channel Index (CCI)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['cci_20'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
            
            # Directional Movement Index
            if all(col in df.columns for col in ['high', 'low', 'close']):
                dm = ta.trend.DMIIndicator(df['high'], df['low'], df['close'], window=14)
                df['di_plus'] = dm.adx_pos()
                df['di_minus'] = dm.adx_neg()
            
            # Parabolic SAR
            if all(col in df.columns for col in ['high', 'low']):
                df['psar'] = ta.trend.PSARIndicator(df['high'], df['low']).psar()
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame, strategy: str = 'sma_crossover') -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with indicators
            strategy: Trading strategy name
            
        Returns:
            DataFrame with trading signals added
        """
        df_signals = df.copy()
        
        if strategy == 'sma_crossover':
            df_signals = self._sma_crossover_signals(df_signals)
        elif strategy == 'rsi_overbought_oversold':
            df_signals = self._rsi_signals(df_signals)
        elif strategy == 'macd_crossover':
            df_signals = self._macd_signals(df_signals)
        elif strategy == 'bollinger_bands':
            df_signals = self._bollinger_bands_signals(df_signals)
        elif strategy == 'combined':
            df_signals = self._combined_signals(df_signals)
        else:
            logger.warning(f"Unknown strategy: {strategy}")
        
        return df_signals
    
    def _sma_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate SMA crossover signals
        
        Args:
            df: DataFrame with SMA indicators
            
        Returns:
            DataFrame with SMA signals
        """
        if 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            logger.error("SMA columns not found")
            return df
        
        # Calculate crossovers
        df['sma_cross_above'] = (df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
        df['sma_cross_below'] = (df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
        
        # Generate signals
        df['signal'] = 0  # Hold
        df.loc[df['sma_cross_above'], 'signal'] = 1  # Buy
        df.loc[df['sma_cross_below'], 'signal'] = -1  # Sell
        
        return df
    
    def _rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RSI-based signals
        
        Args:
            df: DataFrame with RSI indicators
            
        Returns:
            DataFrame with RSI signals
        """
        if 'rsi_14' not in df.columns:
            logger.error("RSI column not found")
            return df
        
        # RSI overbought/oversold signals
        df['rsi_overbought'] = df['rsi_14'] > 70
        df['rsi_oversold'] = df['rsi_14'] < 30
        
        # Generate signals
        df['signal'] = 0  # Hold
        df.loc[df['rsi_oversold'] & (df['rsi_14'].shift(1) >= 30), 'signal'] = 1  # Buy
        df.loc[df['rsi_overbought'] & (df['rsi_14'].shift(1) <= 70), 'signal'] = -1  # Sell
        
        return df
    
    def _macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate MACD crossover signals
        
        Args:
            df: DataFrame with MACD indicators
            
        Returns:
            DataFrame with MACD signals
        """
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            logger.error("MACD columns not found")
            return df
        
        # MACD crossovers
        df['macd_cross_above'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_below'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Generate signals
        df['signal'] = 0  # Hold
        df.loc[df['macd_cross_above'], 'signal'] = 1  # Buy
        df.loc[df['macd_cross_below'], 'signal'] = -1  # Sell
        
        return df
    
    def _bollinger_bands_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Bollinger Bands signals
        
        Args:
            df: DataFrame with Bollinger Bands indicators
            
        Returns:
            DataFrame with Bollinger Bands signals
        """
        if not all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
            logger.error("Bollinger Bands columns not found")
            return df
        
        # Bollinger Bands signals
        df['bb_above_upper'] = df['close'] > df['bb_upper']
        df['bb_below_lower'] = df['close'] < df['bb_lower']
        
        # Generate signals (contrarian approach)
        df['signal'] = 0  # Hold
        df.loc[df['bb_below_lower'] & (df['close'].shift(1) >= df['bb_lower'].shift(1)), 'signal'] = 1  # Buy
        df.loc[df['bb_above_upper'] & (df['close'].shift(1) <= df['bb_upper'].shift(1)), 'signal'] = -1  # Sell
        
        return df
    
    def _combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined signals from multiple indicators
        
        Args:
            df: DataFrame with all indicators
            
        Returns:
            DataFrame with combined signals
        """
        # Initialize signal column
        df['signal'] = 0
        
        # SMA crossover (weight: 2)
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma_signal = np.where(df['sma_20'] > df['sma_50'], 1, -1)
            df['signal'] += sma_signal * 2
        
        # RSI (weight: 1)
        if 'rsi_14' in df.columns:
            rsi_signal = np.where(df['rsi_14'] < 30, 1, np.where(df['rsi_14'] > 70, -1, 0))
            df['signal'] += rsi_signal * 1
        
        # MACD (weight: 1)
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_signal = np.where(df['macd'] > df['macd_signal'], 1, -1)
            df['signal'] += macd_signal * 1
        
        # Normalize signals
        df['signal'] = np.where(df['signal'] > 2, 1, np.where(df['signal'] < -2, -1, 0))
        
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Dictionary with indicator summaries
        """
        summary = {}
        
        # Moving averages summary
        ma_columns = [col for col in df.columns if 'sma_' in col or 'ema_' in col]
        if ma_columns:
            summary['moving_averages'] = {
                'count': len(ma_columns),
                'latest_values': {col: df[col].iloc[-1] if not df[col].empty else None for col in ma_columns}
            }
        
        # Momentum indicators summary
        momentum_columns = ['rsi_14', 'rsi_30', 'macd', 'stoch_k', 'stoch_d']
        momentum_data = {col: df[col].iloc[-1] for col in momentum_columns if col in df.columns and not df[col].empty}
        if momentum_data:
            summary['momentum'] = momentum_data
        
        # Volatility indicators summary
        volatility_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'volatility_20']
        volatility_data = {col: df[col].iloc[-1] for col in volatility_columns if col in df.columns and not df[col].empty}
        if volatility_data:
            summary['volatility'] = volatility_data
        
        # Volume indicators summary
        volume_columns = ['volume_sma_20', 'obv', 'adl']
        volume_data = {col: df[col].iloc[-1] for col in volume_columns if col in df.columns and not df[col].empty}
        if volume_data:
            summary['volume'] = volume_data
        
        # Trend indicators summary
        trend_columns = ['adx', 'aroon_up', 'aroon_down', 'cci_20']
        trend_data = {col: df[col].iloc[-1] for col in trend_columns if col in df.columns and not df[col].empty}
        if trend_data:
            summary['trend'] = trend_data
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import StockDataFetcher
    from data_cleaner import StockDataCleaner
    
    print("Testing technical indicators...")
    
    # Fetch and clean data
    fetcher = StockDataFetcher()
    raw_data = fetcher.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    
    if not raw_data.empty:
        cleaner = StockDataCleaner()
        clean_data, _ = cleaner.clean_data(raw_data, 'AAPL')
        
        # Calculate indicators
        calculator = TechnicalIndicators()
        indicators_data = calculator.calculate_all_indicators(clean_data)
        
        print(f"Original columns: {len(clean_data.columns)}")
        print(f"With indicators: {len(indicators_data.columns)}")
        print(f"New indicators: {len(indicators_data.columns) - len(clean_data.columns)}")
        
        # Generate signals
        signals_data = calculator.generate_trading_signals(indicators_data, 'sma_crossover')
        print(f"Signals generated: {signals_data['signal'].value_counts().to_dict()}")
        
        # Get indicator summary
        summary = calculator.get_indicator_summary(indicators_data)
        print("\nIndicator Summary:")
        for category, data in summary.items():
            print(f"{category}: {data}")
    else:
        print("No data available for testing")
