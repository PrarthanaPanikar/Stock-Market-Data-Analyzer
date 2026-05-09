"""
Backtest Engine Module
Handles strategy backtesting and performance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital for backtest
        """
        self.initial_capital = initial_capital
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005   # 0.05% slippage
        
    def run_backtest(self, df: pd.DataFrame, signal_column: str = 'signal') -> Dict:
        """
        Run backtest on trading signals
        
        Args:
            df: DataFrame with price data and signals
            signal_column: Column name containing trading signals
            
        Returns:
            Dictionary with backtest results
        """
        if df.empty:
            logger.error("Empty DataFrame provided")
            return {"error": "Empty DataFrame"}
        
        if 'close' not in df.columns or signal_column not in df.columns:
            logger.error("Required columns not found")
            return {"error": "Required columns not found"}
        
        logger.info(f"Running backtest with initial capital: ${self.initial_capital:,.2f}")
        
        # Initialize backtracking variables
        capital = self.initial_capital
        position = 0  # Number of shares held
        trades = []
        equity_curve = []
        
        # Track performance metrics
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        max_drawdown = 0
        peak_equity = self.initial_capital
        
        for i in range(len(df)):
            current_date = df.index[i] if hasattr(df.index, 'date') else df.iloc[i]['trading_date']
            current_price = df.iloc[i]['close']
            signal = df.iloc[i][signal_column]
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * current_price)
            equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': capital,
                'position': position,
                'price': current_price
            })
            
            # Update peak equity and drawdown
            if portfolio_value > peak_equity:
                peak_equity = portfolio_value
            
            current_drawdown = (peak_equity - portfolio_value) / peak_equity
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            
            # Process trading signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size (invest all available cash)
                shares_to_buy = capital / (current_price * (1 + self.commission_rate + self.slippage_rate))
                
                if shares_to_buy > 0:
                    trade_cost = shares_to_buy * current_price * (self.commission_rate + self.slippage_rate)
                    capital -= (shares_to_buy * current_price + trade_cost)
                    position = shares_to_buy
                    
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'cost': shares_to_buy * current_price + trade_cost,
                        'signal': signal
                    })
                    total_trades += 1
            
            elif signal == -1 and position > 0:  # Sell signal
                # Sell all position
                sell_value = position * current_price * (1 - self.commission_rate - self.slippage_rate)
                capital += sell_value
                
                # Calculate trade P&L
                buy_trades = [t for t in trades if t['action'] == 'BUY' and 'buy_price' not in t]
                if buy_trades:
                    last_buy = buy_trades[-1]
                    last_buy['buy_price'] = last_buy['price']
                    pnl = sell_value - last_buy['cost']
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': position,
                        'proceeds': sell_value,
                        'pnl': pnl,
                        'signal': signal
                    })
                
                position = 0
                total_trades += 1
        
        # Calculate final portfolio value
        final_value = capital + (position * df.iloc[-1]['close'])
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate performance metrics
        returns = equity_df['portfolio_value'].pct_change().dropna()
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': (final_value - self.initial_capital) / self.initial_capital,
            'annualized_return': self._calculate_annualized_return(equity_df),
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'profit_factor': self._calculate_profit_factor(trades),
            'average_trade': self._calculate_average_trade(trades),
            'equity_curve': equity_df,
            'trades': trades
        }
        
        logger.info(f"Backtest completed. Total return: {metrics['total_return']:.2%}")
        return metrics
    
    def _calculate_annualized_return(self, equity_df: pd.DataFrame) -> float:
        """
        Calculate annualized return
        
        Args:
            equity_df: DataFrame with portfolio values over time
            
        Returns:
            Annualized return
        """
        if len(equity_df) < 2:
            return 0.0
        
        start_value = equity_df['portfolio_value'].iloc[0]
        end_value = equity_df['portfolio_value'].iloc[-1]
        
        # Calculate days
        if hasattr(equity_df.index, 'date'):
            days = (equity_df.index[-1] - equity_df.index[0]).days
        else:
            # Fallback for non-datetime index
            days = len(equity_df)
        
        if days <= 0:
            return 0.0
        
        # Calculate annualized return
        total_return = (end_value - start_value) / start_value
        years = days / 365.25
        
        if years <= 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        return sharpe_ratio
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Args:
            trades: List of completed trades
            
        Returns:
            Profit factor
        """
        gross_profit = 0
        gross_loss = 0
        
        for trade in trades:
            if 'pnl' in trade:
                if trade['pnl'] > 0:
                    gross_profit += trade['pnl']
                elif trade['pnl'] < 0:
                    gross_loss += abs(trade['pnl'])
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_average_trade(self, trades: List[Dict]) -> float:
        """
        Calculate average trade P&L
        
        Args:
            trades: List of completed trades
            
        Returns:
            Average trade P&L
        """
        pnl_trades = [trade['pnl'] for trade in trades if 'pnl' in trade]
        
        if not pnl_trades:
            return 0.0
        
        return np.mean(pnl_trades)
    
    def compare_strategies(self, df: pd.DataFrame, strategies: List[str]) -> Dict:
        """
        Compare multiple trading strategies
        
        Args:
            df: DataFrame with price data and multiple signal columns
            strategies: List of strategy names (signal column names)
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for strategy in strategies:
            if strategy in df.columns:
                logger.info(f"Backtesting strategy: {strategy}")
                results = self.run_backtest(df, strategy)
                comparison[strategy] = results
            else:
                logger.warning(f"Strategy column '{strategy}' not found")
        
        # Create comparison table
        comparison_table = self._create_comparison_table(comparison)
        
        return {
            'individual_results': comparison,
            'comparison_table': comparison_table,
            'best_strategy': self._find_best_strategy(comparison_table)
        }
    
    def _create_comparison_table(self, comparison: Dict) -> pd.DataFrame:
        """
        Create comparison table for strategies
        
        Args:
            comparison: Dictionary with strategy results
            
        Returns:
            DataFrame with comparison metrics
        """
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        table_data = []
        
        for strategy, results in comparison.items():
            if 'error' not in results:
                row = {'strategy': strategy}
                for metric in metrics:
                    row[metric] = results.get(metric, 0)
                table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def _find_best_strategy(self, comparison_table: pd.DataFrame) -> Dict:
        """
        Find best strategy based on multiple metrics
        
        Args:
            comparison_table: DataFrame with strategy comparison
            
        Returns:
            Dictionary with best strategy information
        """
        if comparison_table.empty:
            return {"error": "No strategies to compare"}
        
        # Calculate composite score (weighted average of normalized metrics)
        weights = {
            'total_return': 0.3,
            'sharpe_ratio': 0.3,
            'max_drawdown': -0.2,  # Negative because lower is better
            'win_rate': 0.2
        }
        
        scores = []
        for _, row in comparison_table.iterrows():
            score = 0
            for metric, weight in weights.items():
                if metric in row and not pd.isna(row[metric]):
                    # Normalize metric (simple min-max scaling)
                    if comparison_table[metric].std() > 0:
                        normalized = (row[metric] - comparison_table[metric].min()) / comparison_table[metric].std()
                        score += normalized * weight
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_strategy = comparison_table.iloc[best_idx]['strategy']
        
        return {
            'strategy': best_strategy,
            'score': scores[best_idx],
            'metrics': comparison_table[comparison_table['strategy'] == best_strategy].iloc[0].to_dict()
        }
    
    def generate_backtest_report(self, results: Dict) -> str:
        """
        Generate human-readable backtest report
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Formatted report string
        """
        if 'error' in results:
            return f"Backtest Error: {results['error']}"
        
        report = []
        report.append("BACKTEST REPORT")
        report.append("=" * 50)
        report.append(f"Initial Capital: ${results['initial_capital']:,.2f}")
        report.append(f"Final Value: ${results['final_value']:,.2f}")
        report.append(f"Total Return: {results['total_return']:.2%}")
        report.append(f"Annualized Return: {results['annualized_return']:.2%}")
        report.append(f"Volatility: {results['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        report.append("")
        report.append("TRADING STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Trades: {results['total_trades']}")
        report.append(f"Winning Trades: {results['winning_trades']}")
        report.append(f"Losing Trades: {results['losing_trades']}")
        report.append(f"Win Rate: {results['win_rate']:.2%}")
        report.append(f"Profit Factor: {results['profit_factor']:.2f}")
        report.append(f"Average Trade: ${results['average_trade']:,.2f}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import StockDataFetcher
    from data_cleaner import StockDataCleaner
    from technical_indicators import TechnicalIndicators
    
    print("Testing backtest engine...")
    
    # Fetch, clean, and calculate indicators
    fetcher = StockDataFetcher()
    raw_data = fetcher.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    
    if not raw_data.empty:
        cleaner = StockDataCleaner()
        clean_data, _ = cleaner.clean_data(raw_data, 'AAPL')
        
        calculator = TechnicalIndicators()
        indicators_data = calculator.calculate_all_indicators(clean_data)
        
        # Generate signals
        signals_data = calculator.generate_trading_signals(indicators_data, 'sma_crossover')
        
        # Run backtest
        backtest = BacktestEngine(initial_capital=100000)
        results = backtest.run_backtest(signals_data, 'signal')
        
        print("\nBacktest Results:")
        print(backtest.generate_backtest_report(results))
        
        # Test multiple strategies
        signals_data_rsi = calculator.generate_trading_signals(indicators_data, 'rsi_overbought_oversold')
        signals_data_macd = calculator.generate_trading_signals(indicators_data, 'macd_crossover')
        
        # Combine signals for comparison
        comparison_data = signals_data.copy()
        comparison_data['rsi_signal'] = signals_data_rsi['signal']
        comparison_data['macd_signal'] = signals_data_macd['signal']
        
        comparison_results = backtest.compare_strategies(comparison_data, ['signal', 'rsi_signal', 'macd_signal'])
        
        print("\nStrategy Comparison:")
        if 'comparison_table' in comparison_results:
            print(comparison_results['comparison_table'].to_string())
        
        if 'best_strategy' in comparison_results:
            print(f"\nBest Strategy: {comparison_results['best_strategy']}")
    else:
        print("No data available for testing")
