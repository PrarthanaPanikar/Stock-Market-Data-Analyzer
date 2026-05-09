"""
Stock Market Data Analyzer - Main Application
Comprehensive stock market analysis tool with technical indicators, backtesting, and visualization
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import StockDataFetcher
from src.data_cleaner import StockDataCleaner
from src.technical_indicators import TechnicalIndicators
from src.backtest_engine import BacktestEngine
from src.visualizer import StockVisualizer
from src.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockMarketAnalyzer:
    """
    Main application class for stock market analysis
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the analyzer
        
        Args:
            initial_capital: Initial capital for backtesting
        """
        self.fetcher = StockDataFetcher()
        self.cleaner = StockDataCleaner()
        self.indicators = TechnicalIndicators()
        self.backtest = BacktestEngine(initial_capital)
        self.visualizer = StockVisualizer()
        self.reporter = ReportGenerator()
        
        logger.info("Stock Market Analyzer initialized")
    
    def analyze_stock(self, symbol: str, start_date: str, end_date: str = None, 
                    strategy: str = 'sma_crossover', generate_report: bool = True) -> dict:
        """
        Perform comprehensive stock analysis
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            strategy: Trading strategy to backtest
            generate_report: Whether to generate PDF report
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting analysis for {symbol}")
        
        try:
            # Step 1: Fetch data
            logger.info("Step 1: Fetching stock data...")
            raw_data = self.fetcher.fetch_data(symbol, start_date, end_date)
            
            if raw_data.empty:
                return {"error": f"No data found for {symbol}"}
            
            # Step 2: Clean data
            logger.info("Step 2: Cleaning data...")
            clean_data, cleaning_report = self.cleaner.clean_data(raw_data, symbol)
            
            # Step 3: Calculate technical indicators
            logger.info("Step 3: Calculating technical indicators...")
            indicators_data = self.indicators.calculate_all_indicators(clean_data)
            
            # Step 4: Generate trading signals
            logger.info("Step 4: Generating trading signals...")
            signals_data = self.indicators.generate_trading_signals(indicators_data, strategy)
            
            # Step 5: Run backtest
            logger.info("Step 5: Running backtest...")
            backtest_results = self.backtest.run_backtest(signals_data, 'signal')
            
            # Step 6: Create visualizations
            logger.info("Step 6: Creating visualizations...")
            self._create_visualizations(indicators_data, backtest_results, symbol)
            
            # Step 7: Generate report
            report_path = None
            if generate_report:
                logger.info("Step 7: Generating report...")
                indicators_summary = self.indicators.get_indicator_summary(indicators_data)
                report_path = self.reporter.generate_comprehensive_report(
                    indicators_data, symbol, backtest_results, indicators_summary
                )
            
            # Compile results
            results = {
                'symbol': symbol,
                'data_period': {
                    'start_date': start_date,
                    'end_date': end_date or datetime.now().strftime('%Y-%m-%d'),
                    'data_points': len(clean_data)
                },
                'data_quality': {
                    'cleaning_report': cleaning_report,
                    'quality_score': cleaning_report.get('data_quality_score', 0)
                },
                'indicators': {
                    'summary': self.indicators.get_indicator_summary(indicators_data),
                    'signals_count': signals_data['signal'].value_counts().to_dict()
                },
                'backtest': backtest_results,
                'visualizations': {
                    'price_chart': f"../outputs/{symbol}_price_chart.html",
                    'dashboard': f"../outputs/{symbol}_dashboard.html",
                    'backtest_chart': f"../outputs/{symbol}_backtest.html"
                },
                'report': report_path
            }
            
            logger.info(f"Analysis completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def _create_visualizations(self, indicators_data: pd.DataFrame, backtest_results: dict, symbol: str):
        """
        Create all visualizations for the analysis
        
        Args:
            indicators_data: DataFrame with indicators
            backtest_results: Backtest results
            symbol: Stock symbol
        """
        try:
            # Price chart
            self.visualizer.create_price_chart(indicators_data, symbol)
            
            # Dashboard
            self.visualizer.create_summary_dashboard(indicators_data, symbol)
            
            # Backtest chart
            if 'equity_curve' in backtest_results and 'trades' in backtest_results:
                self.visualizer.create_backtest_chart(
                    backtest_results['equity_curve'], 
                    backtest_results['trades'], 
                    symbol
                )
            
            # Returns analysis
            if 'returns' in indicators_data.columns:
                self.visualizer.create_returns_analysis(indicators_data, symbol)
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def compare_strategies(self, symbol: str, start_date: str, end_date: str = None,
                         strategies: list = None) -> dict:
        """
        Compare multiple trading strategies
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            strategies: List of strategies to compare
            
        Returns:
            Dictionary with comparison results
        """
        if strategies is None:
            strategies = ['sma_crossover', 'rsi_overbought_oversold', 'macd_crossover']
        
        logger.info(f"Comparing strategies for {symbol}: {strategies}")
        
        try:
            # Fetch and prepare data
            raw_data = self.fetcher.fetch_data(symbol, start_date, end_date)
            if raw_data.empty:
                return {"error": f"No data found for {symbol}"}
            
            clean_data, _ = self.cleaner.clean_data(raw_data, symbol)
            indicators_data = self.indicators.calculate_all_indicators(clean_data)
            
            # Generate signals for all strategies
            comparison_data = indicators_data.copy()
            for strategy in strategies:
                signals_data = self.indicators.generate_trading_signals(indicators_data, strategy)
                comparison_data[f'{strategy}_signal'] = signals_data['signal']
            
            # Compare strategies
            comparison_results = self.backtest.compare_strategies(comparison_data, [f'{s}_signal' for s in strategies])
            
            # Create comparison chart
            if 'individual_results' in comparison_results:
                self.visualizer.create_performance_comparison(comparison_results['individual_results'])
            
            logger.info("Strategy comparison completed")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
            return {"error": str(e)}
    
    def batch_analyze(self, symbols: list, start_date: str, end_date: str = None) -> dict:
        """
        Analyze multiple stocks in batch
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with batch analysis results
        """
        logger.info(f"Starting batch analysis for {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"Analyzing {symbol}...")
            result = self.analyze_stock(symbol, start_date, end_date, generate_report=False)
            results[symbol] = result
        
        # Create correlation analysis if we have multiple results
        if len(results) > 1:
            try:
                self._create_correlation_analysis(symbols, start_date, end_date)
            except Exception as e:
                logger.error(f"Error creating correlation analysis: {str(e)}")
        
        logger.info("Batch analysis completed")
        return results
    
    def _create_correlation_analysis(self, symbols: list, start_date: str, end_date: str):
        """
        Create correlation analysis for multiple stocks
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
        """
        # Fetch data for all symbols
        returns_data = pd.DataFrame()
        
        for symbol in symbols:
            raw_data = self.fetcher.fetch_data(symbol, start_date, end_date)
            if not raw_data.empty:
                clean_data, _ = self.cleaner.clean_data(raw_data, symbol)
                clean_data['returns'] = clean_data['close'].pct_change()
                returns_data[symbol] = clean_data['returns']
        
        if not returns_data.empty:
            # Create correlation heatmap
            self.visualizer.create_correlation_heatmap(returns_data, symbols)
            logger.info("Correlation analysis created")

def main():
    """
    Main function for command-line interface
    """
    parser = argparse.ArgumentParser(description='Stock Market Data Analyzer')
    parser.add_argument('--symbol', '-s', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start-date', '-sd', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-ed', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', '-st', type=str, default='sma_crossover', 
                       choices=['sma_crossover', 'rsi_overbought_oversold', 'macd_crossover', 'bollinger_bands', 'combined'],
                       help='Trading strategy to use')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare multiple strategies')
    parser.add_argument('--batch', '-b', type=str, nargs='+', help='Batch analyze multiple symbols')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital for backtesting')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = StockMarketAnalyzer(args.capital)
    
    if args.batch:
        # Batch analysis
        results = analyzer.batch_analyze(args.batch, args.start_date, args.end_date)
        print("\nBatch Analysis Results:")
        for symbol, result in results.items():
            if 'error' not in result:
                print(f"{symbol}: Success")
            else:
                print(f"{symbol}: {result['error']}")
    
    elif args.compare:
        # Strategy comparison
        results = analyzer.compare_strategies(args.symbol, args.start_date, args.end_date)
        if 'comparison_table' in results:
            print("\nStrategy Comparison Results:")
            print(results['comparison_table'].to_string())
        if 'best_strategy' in results:
            print(f"\nBest Strategy: {results['best_strategy']}")
    
    else:
        # Single stock analysis
        results = analyzer.analyze_stock(
            args.symbol, 
            args.start_date, 
            args.end_date, 
            args.strategy, 
            not args.no_report
        )
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"\nAnalysis completed for {results['symbol']}")
            print(f"Data Quality Score: {results['data_quality']['quality_score']:.2f}/100")
            
            if 'backtest' in results and 'total_return' in results['backtest']:
                print(f"Strategy Return: {results['backtest']['total_return']:.2%}")
                print(f"Sharpe Ratio: {results['backtest']['sharpe_ratio']:.2f}")
            
            if results.get('report'):
                print(f"Report generated: {results['report']}")
            
            print("\nGenerated files:")
            for viz_type, filepath in results['visualizations'].items():
                print(f"  {viz_type}: {filepath}")

if __name__ == "__main__":
    main()
