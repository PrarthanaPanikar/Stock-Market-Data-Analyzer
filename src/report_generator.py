"""
Report Generator Module
Creates comprehensive analysis reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import os
from jinja2 import Template
import weasyprint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Comprehensive report generator for stock market analysis
    """
    
    def __init__(self, output_dir: str = "../reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created reports directory: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                   df: pd.DataFrame, 
                                   symbol: str, 
                                   backtest_results: Dict = None,
                                   indicators_summary: Dict = None) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            df: DataFrame with analysis data
            symbol: Stock symbol
            backtest_results: Backtest results dictionary
            indicators_summary: Technical indicators summary
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating comprehensive report for {symbol}")
        
        # Extract analysis components
        basic_stats = self._calculate_basic_statistics(df)
        risk_metrics = self._calculate_risk_metrics(df)
        performance_metrics = backtest_results if backtest_results else {}
        
        # Generate report content
        report_data = {
            'symbol': symbol,
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_period': self._get_analysis_period(df),
            'basic_stats': basic_stats,
            'risk_metrics': risk_metrics,
            'performance_metrics': performance_metrics,
            'indicators_summary': indicators_summary or {},
            'recommendations': self._generate_recommendations(df, backtest_results),
            'data_quality': self._assess_data_quality(df)
        }
        
        # Generate HTML report
        html_content = self._generate_html_report(report_data)
        
        # Save HTML report
        html_path = os.path.join(self.output_dir, f"{symbol}_analysis_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate PDF report
        try:
            pdf_path = os.path.join(self.output_dir, f"{symbol}_analysis_report.pdf")
            weasyprint.HTML(string=html_content).write_pdf(pdf_path)
            logger.info(f"PDF report saved: {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not generate PDF report: {str(e)}")
            pdf_path = None
        
        logger.info(f"HTML report saved: {html_path}")
        return html_path
    
    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate basic statistical measures
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with basic statistics
        """
        if df.empty or 'close' not in df.columns:
            return {}
        
        close_prices = df['close']
        
        stats = {
            'current_price': close_prices.iloc[-1] if not close_prices.empty else 0,
            'period_high': close_prices.max(),
            'period_low': close_prices.min(),
            'average_price': close_prices.mean(),
            'price_change': close_prices.iloc[-1] - close_prices.iloc[0] if len(close_prices) > 1 else 0,
            'price_change_pct': ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100 if len(close_prices) > 1 else 0,
            'volatility': close_prices.std(),
            'data_points': len(close_prices)
        }
        
        # Add volume statistics if available
        if 'volume' in df.columns:
            volume = df['volume'].dropna()
            if not volume.empty:
                stats.update({
                    'average_volume': volume.mean(),
                    'max_volume': volume.max(),
                    'min_volume': volume.min()
                })
        
        return stats
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate risk-related metrics
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with risk metrics
        """
        if df.empty or 'close' not in df.columns:
            return {}
        
        # Calculate daily returns
        returns = df['close'].pct_change().dropna()
        
        if returns.empty:
            return {}
        
        risk_metrics = {
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'max_daily_loss': returns.min(),
            'max_daily_gain': returns.max(),
            'var_95': returns.quantile(0.05),  # 5% VaR
            'var_99': returns.quantile(0.01),  # 1% VaR
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        risk_metrics['max_drawdown'] = drawdown.min()
        risk_metrics['max_drawdown_duration'] = self._calculate_drawdown_duration(drawdown)
        
        return risk_metrics
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """
        Calculate maximum drawdown duration in days
        
        Args:
            drawdown: Drawdown series
            
        Returns:
            Maximum drawdown duration
        """
        is_drawdown = drawdown < 0
        drawdown_periods = is_drawdown.astype(int).groupby((~is_drawdown).cumsum()).cumsum()
        return drawdown_periods.max() if not drawdown_periods.empty else 0
    
    def _get_analysis_period(self, df: pd.DataFrame) -> Dict:
        """
        Get analysis period information
        
        Args:
            df: DataFrame with date index
            
        Returns:
            Dictionary with period information
        """
        if df.empty:
            return {}
        
        start_date = df.index[0] if hasattr(df.index, 'date') else df.iloc[0].get('trading_date')
        end_date = df.index[-1] if hasattr(df.index, 'date') else df.iloc[-1].get('trading_date')
        
        return {
            'start_date': str(start_date),
            'end_date': str(end_date),
            'trading_days': len(df)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, backtest_results: Dict = None) -> List[str]:
        """
        Generate investment recommendations based on analysis
        
        Args:
            df: DataFrame with analysis data
            backtest_results: Backtest results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if df.empty:
            return ["Insufficient data for recommendations"]
        
        # Price-based recommendations
        if 'close' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                recommendations.append("Price is above both short and long-term moving averages - Bullish signal")
            elif current_price < sma_20 < sma_50:
                recommendations.append("Price is below both short and long-term moving averages - Bearish signal")
            else:
                recommendations.append("Price is between moving averages - Neutral/Sideways trend")
        
        # RSI-based recommendations
        if 'rsi_14' in df.columns:
            current_rsi = df['rsi_14'].iloc[-1]
            if current_rsi > 70:
                recommendations.append("RSI indicates overbought conditions - Consider taking profits")
            elif current_rsi < 30:
                recommendations.append("RSI indicates oversold conditions - Potential buying opportunity")
            else:
                recommendations.append("RSI is in neutral range - No clear signal")
        
        # Volatility-based recommendations
        if 'volatility_20' in df.columns:
            current_vol = df['volatility_20'].iloc[-1]
            if current_vol > 0.3:  # 30% annualized volatility
                recommendations.append("High volatility detected - Use smaller position sizes")
            elif current_vol < 0.15:  # 15% annualized volatility
                recommendations.append("Low volatility environment - Good for income strategies")
        
        # Backtest-based recommendations
        if backtest_results and 'total_return' in backtest_results:
            if backtest_results['total_return'] > 0.1:  # 10% return
                recommendations.append("Strategy shows positive returns - Consider systematic implementation")
            elif backtest_results['total_return'] < -0.05:  # -5% return
                recommendations.append("Strategy shows negative returns - Review and optimize before implementation")
        
        # Risk management recommendations
        if 'max_drawdown' in backtest_results and backtest_results['max_drawdown'] > 0.2:
            recommendations.append("High maximum drawdown detected - Implement stricter risk management")
        
        if not recommendations:
            recommendations.append("Continue monitoring for clearer signals")
        
        return recommendations
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Assess data quality metrics
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {'score': 0, 'issues': ['No data available']}
        
        quality_score = 100
        issues = []
        
        # Check for missing data
        missing_data = df.isnull().sum().sum()
        total_data = len(df) * len(df.columns)
        missing_ratio = missing_data / total_data if total_data > 0 else 0
        
        if missing_ratio > 0.05:  # More than 5% missing data
            quality_score -= 20
            issues.append(f"High missing data ratio: {missing_ratio:.2%}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_score -= 10
            issues.append(f"Found {duplicates} duplicate rows")
        
        # Check data consistency
        if all(col in df.columns for col in ['high', 'low', 'close']):
            price_violations = (df['high'] < df['low']).sum()
            if price_violations > 0:
                quality_score -= 15
                issues.append(f"Found {price_violations} price relationship violations")
        
        # Check data freshness
        if hasattr(df.index, 'max'):
            latest_date = df.index.max()
            days_old = (datetime.now().date() - latest_date.date()).days if hasattr(latest_date, 'date') else 0
            if days_old > 7:
                quality_score -= 10
                issues.append(f"Data is {days_old} days old")
        
        return {
            'score': max(0, quality_score),
            'issues': issues,
            'data_points': len(df),
            'columns': len(df.columns),
            'date_range': f"{df.index[0]} to {df.index[-1]}" if hasattr(df.index, 'date') else "Unknown"
        }
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """
        Generate HTML report from template
        
        Args:
            report_data: Dictionary with all report data
            
        Returns:
            HTML content string
        """
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ symbol }} Stock Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { font-size: 1.2em; color: #333; }
        .positive { color: green; }
        .negative { color: red; }
        .neutral { color: #666; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .recommendation { background-color: #e7f3ff; padding: 10px; margin: 5px 0; border-left: 4px solid #2196F3; }
        .warning { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ symbol }} Stock Analysis Report</h1>
        <p>Generated on: {{ date_generated }}</p>
        <p>Analysis Period: {{ analysis_period.start_date }} to {{ analysis_period.end_date }} ({{ analysis_period.trading_days }} trading days)</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${{ "%.2f"|format(basic_stats.current_price) }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Period Change</div>
            <div class="metric-value {{ 'positive' if basic_stats.price_change_pct > 0 else 'negative' if basic_stats.price_change_pct < 0 else 'neutral' }}">
                {{ "%.2f"|format(basic_stats.price_change_pct) }}%
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Volatility</div>
            <div class="metric-value">{{ "%.2f"|format(risk_metrics.annualized_volatility * 100) }}%</div>
        </div>
    </div>

    <div class="section">
        <h2>Basic Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Current Price</td><td>${{ "%.2f"|format(basic_stats.current_price) }}</td></tr>
            <tr><td>Period High</td><td>${{ "%.2f"|format(basic_stats.period_high) }}</td></tr>
            <tr><td>Period Low</td><td>${{ "%.2f"|format(basic_stats.period_low) }}</td></tr>
            <tr><td>Average Price</td><td>${{ "%.2f"|format(basic_stats.average_price) }}</td></tr>
            <tr><td>Price Change</td><td>${{ "%.2f"|format(basic_stats.price_change) }} ({{ "%.2f"|format(basic_stats.price_change_pct) }}%)</td></tr>
            {% if basic_stats.average_volume %}
            <tr><td>Average Volume</td><td>{{ "{:,}".format(basic_stats.average_volume|round) }}</td></tr>
            {% endif %}
        </table>
    </div>

    <div class="section">
        <h2>Risk Analysis</h2>
        <table>
            <tr><th>Risk Metric</th><th>Value</th></tr>
            <tr><td>Daily Volatility</td><td>{{ "%.2f"|format(risk_metrics.daily_volatility * 100) }}%</td></tr>
            <tr><td>Annualized Volatility</td><td>{{ "%.2f"|format(risk_metrics.annualized_volatility * 100) }}%</td></tr>
            <tr><td>Maximum Drawdown</td><td>{{ "%.2f"|format(risk_metrics.max_drawdown * 100) }}%</td></tr>
            <tr><td>95% VaR (Daily)</td><td>{{ "%.2f"|format(risk_metrics.var_95 * 100) }}%</td></tr>
            <tr><td>99% VaR (Daily)</td><td>{{ "%.2f"|format(risk_metrics.var_99 * 100) }}%</td></tr>
            <tr><td>Skewness</td><td>{{ "%.3f"|format(risk_metrics.skewness) }}</td></tr>
            <tr><td>Kurtosis</td><td>{{ "%.3f"|format(risk_metrics.kurtosis) }}</td></tr>
        </table>
    </div>

    {% if performance_metrics %}
    <div class="section">
        <h2>Strategy Performance</h2>
        <table>
            <tr><th>Performance Metric</th><th>Value</th></tr>
            <tr><td>Total Return</td><td>{{ "%.2f"|format(performance_metrics.total_return * 100) }}%</td></tr>
            <tr><td>Annualized Return</td><td>{{ "%.2f"|format(performance_metrics.annualized_return * 100) }}%</td></tr>
            <tr><td>Sharpe Ratio</td><td>{{ "%.2f"|format(performance_metrics.sharpe_ratio) }}</td></tr>
            <tr><td>Win Rate</td><td>{{ "%.2f"|format(performance_metrics.win_rate * 100) }}%</td></tr>
            <tr><td>Total Trades</td><td>{{ performance_metrics.total_trades }}</td></tr>
            <tr><td>Profit Factor</td><td>{{ "%.2f"|format(performance_metrics.profit_factor) }}</td></tr>
        </table>
    </div>
    {% endif %}

    <div class="section">
        <h2>Investment Recommendations</h2>
        {% for rec in recommendations %}
        <div class="recommendation">{{ rec }}</div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Data Quality Assessment</h2>
        <div class="metric">
            <div class="metric-label">Quality Score</div>
            <div class="metric-value">{{ data_quality.score }}/100</div>
        </div>
        {% if data_quality.issues %}
            {% for issue in data_quality.issues %}
            <div class="warning">{{ issue }}</div>
            {% endfor %}
        {% else %}
        <p>No data quality issues detected.</p>
        {% endif %}
        <p>Data Points: {{ data_quality.data_points }}, Columns: {{ data_quality.columns }}</p>
        <p>Date Range: {{ data_quality.date_range }}</p>
    </div>

    <div class="section">
        <h2>Disclaimer</h2>
        <p><em>This report is for educational purposes only and should not be considered as financial advice. 
        Past performance does not guarantee future results. Always conduct your own research before making investment decisions.</em></p>
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        return template.render(**report_data)
    
    def generate_summary_report(self, results: Dict) -> str:
        """
        Generate a brief summary report
        
        Args:
            results: Dictionary with analysis results
            
        Returns:
            Summary report string
        """
        summary = []
        summary.append("STOCK MARKET ANALYSIS SUMMARY")
        summary.append("=" * 40)
        
        if 'symbol' in results:
            summary.append(f"Symbol: {results['symbol']}")
        
        if 'basic_stats' in results:
            stats = results['basic_stats']
            summary.append(f"Current Price: ${stats.get('current_price', 0):.2f}")
            summary.append(f"Period Change: {stats.get('price_change_pct', 0):.2f}%")
        
        if 'risk_metrics' in results:
            risk = results['risk_metrics']
            summary.append(f"Volatility: {risk.get('annualized_volatility', 0):.2%}")
            summary.append(f"Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
        
        if 'performance_metrics' in results:
            perf = results['performance_metrics']
            summary.append(f"Strategy Return: {perf.get('total_return', 0):.2%}")
            summary.append(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        
        if 'recommendations' in results:
            summary.append("\nKey Recommendations:")
            for rec in results['recommendations'][:3]:  # Top 3 recommendations
                summary.append(f"• {rec}")
        
        return "\n".join(summary)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import StockDataFetcher
    from data_cleaner import StockDataCleaner
    from technical_indicators import TechnicalIndicators
    from backtest_engine import BacktestEngine
    
    print("Testing report generator...")
    
    # Fetch, clean, and analyze data
    fetcher = StockDataFetcher()
    raw_data = fetcher.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    
    if not raw_data.empty:
        cleaner = StockDataCleaner()
        clean_data, _ = cleaner.clean_data(raw_data, 'AAPL')
        
        calculator = TechnicalIndicators()
        indicators_data = calculator.calculate_all_indicators(clean_data)
        
        # Generate signals and backtest
        signals_data = calculator.generate_trading_signals(indicators_data, 'sma_crossover')
        backtest = BacktestEngine()
        backtest_results = backtest.run_backtest(signals_data, 'signal')
        
        # Get indicators summary
        indicators_summary = calculator.get_indicator_summary(indicators_data)
        
        # Generate report
        generator = ReportGenerator()
        report_path = generator.generate_comprehensive_report(
            indicators_data, 
            'AAPL', 
            backtest_results, 
            indicators_summary
        )
        
        print(f"Report generated: {report_path}")
        
        # Generate summary
        summary_data = {
            'symbol': 'AAPL',
            'basic_stats': generator._calculate_basic_statistics(indicators_data),
            'risk_metrics': generator._calculate_risk_metrics(indicators_data),
            'performance_metrics': backtest_results,
            'recommendations': generator._generate_recommendations(indicators_data, backtest_results)
        }
        
        summary = generator.generate_summary_report(summary_data)
        print("\nSummary Report:")
        print(summary)
    else:
        print("No data available for testing")
