"""
Visualizer Module
Handles data visualization and chart generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockVisualizer:
    """
    Comprehensive visualization toolkit for stock market data
    """
    
    def __init__(self, output_dir: str = "../outputs"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = output_dir
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def create_price_chart(self, df: pd.DataFrame, symbol: str = "", save: bool = True) -> go.Figure:
        """
        Create interactive price chart with moving averages
        
        Args:
            df: DataFrame with price data
            symbol: Stock symbol for title
            save: Whether to save the chart
            
        Returns:
            Plotly figure object
        """
        if df.empty or 'close' not in df.columns:
            logger.error("Invalid data for price chart")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_width=[0.2, 0.7]
        )
        
        # Add candlestick or line chart
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Add moving averages if available
        ma_columns = [col for col in df.columns if 'sma_' in col or 'ema_' in col]
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, ma_col in enumerate(ma_columns[:6]):  # Limit to 6 MAs
            if ma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ma_col],
                        mode='lines',
                        name=ma_col.upper(),
                        line=dict(color=colors[i % len(colors)], width=1)
                    ),
                    row=1, col=1
                )
        
        # Add volume if available
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        # Save chart if requested
        if save:
            self.save_plotly_chart(fig, f'{symbol}_price_chart')
        
        return fig
    
    def create_indicator_chart(self, df: pd.DataFrame, indicators: List[str], symbol: str = "", save: bool = True) -> go.Figure:
        """
        Create chart with technical indicators
        
        Args:
            df: DataFrame with indicators
            indicators: List of indicator columns to plot
            symbol: Stock symbol for title
            save: Whether to save the chart
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            logger.error("Empty DataFrame for indicator chart")
            return go.Figure()
        
        # Create subplots based on indicator types
        price_indicators = ['close', 'sma_20', 'sma_50', 'ema_20', 'ema_50', 'bb_upper', 'bb_middle', 'bb_lower']
        momentum_indicators = ['rsi_14', 'rsi_30', 'macd', 'macd_signal', 'stoch_k', 'stoch_d']
        volume_indicators = ['volume', 'obv', 'adl']
        
        # Determine subplot layout
        subplot_count = 1
        if any(ind in momentum_indicators for ind in indicators):
            subplot_count += 1
        if any(ind in volume_indicators for ind in indicators):
            subplot_count += 1
        
        # Create subplots
        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Price & Moving Averages', 'Momentum Indicators', 'Volume Indicators'][:subplot_count]
        )
        
        current_row = 1
        
        # Plot price and moving averages
        if 'close' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='blue')),
                row=current_row, col=1
            )
        
        # Plot selected indicators
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        color_idx = 0
        
        for indicator in indicators:
            if indicator in df.columns:
                if indicator in price_indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, 
                            y=df[indicator], 
                            name=indicator.upper(),
                            line=dict(color=colors[color_idx % len(colors)])
                        ),
                        row=current_row, col=1
                    )
                    color_idx += 1
        
        # Plot momentum indicators
        if any(ind in momentum_indicators for ind in indicators):
            current_row += 1
            color_idx = 0
            
            for indicator in indicators:
                if indicator in momentum_indicators and indicator in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[indicator],
                            name=indicator.upper(),
                            line=dict(color=colors[color_idx % len(colors)])
                        ),
                        row=current_row, col=1
                    )
                    color_idx += 1
        
        # Plot volume indicators
        if any(ind in volume_indicators for ind in indicators):
            current_row += 1
            color_idx = 0
            
            for indicator in indicators:
                if indicator in volume_indicators and indicator in df.columns:
                    if indicator == 'volume':
                        fig.add_trace(
                            go.Bar(x=df.index, y=df[indicator], name=indicator.upper()),
                            row=current_row, col=1
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[indicator],
                                name=indicator.upper(),
                                line=dict(color=colors[color_idx % len(colors)])
                            ),
                            row=current_row, col=1
                        )
                    color_idx += 1
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Indicators',
            height=300 * subplot_count,
            showlegend=True
        )
        
        # Save chart if requested
        if save:
            self.save_plotly_chart(fig, f'{symbol}_indicators')
        
        return fig
    
    def create_returns_analysis(self, df: pd.DataFrame, symbol: str = "", save: bool = True) -> go.Figure:
        """
        Create returns analysis charts
        
        Args:
            df: DataFrame with returns data
            symbol: Stock symbol for title
            save: Whether to save the chart
            
        Returns:
            Plotly figure object
        """
        if df.empty or 'returns' not in df.columns:
            logger.error("Returns data not found")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 'Returns Distribution', 'Rolling Volatility', 'Drawdown'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calculate metrics
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        df['rolling_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['cumulative_max'] = df['cumulative_returns'].cummax()
        df['drawdown'] = df['cumulative_returns'] - df['cumulative_max']
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(x=df.index, y=df['cumulative_returns'], name='Cumulative Returns'),
            row=1, col=1
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=df['returns'], name='Returns Distribution', nbinsx=50),
            row=1, col=2
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rolling_volatility'], name='Rolling Volatility'),
            row=2, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=df.index, y=df['drawdown'], name='Drawdown', fill='tonexty'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Returns Analysis',
            height=800,
            showlegend=True
        )
        
        # Save chart if requested
        if save:
            self.save_plotly_chart(fig, f'{symbol}_returns_analysis')
        
        return fig
    
    def create_backtest_chart(self, equity_curve: pd.DataFrame, trades: List[Dict], symbol: str = "", save: bool = True) -> go.Figure:
        """
        Create backtest visualization
        
        Args:
            equity_curve: DataFrame with portfolio values over time
            trades: List of trade dictionaries
            symbol: Stock symbol for title
            save: Whether to save the chart
            
        Returns:
            Plotly figure object
        """
        if equity_curve.empty:
            logger.error("Empty equity curve data")
            return go.Figure()
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Backtest Results', 'Trades'),
            row_heights=[0.7, 0.3]
        )
        
        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add trade markers
        if trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        name='Buy',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ),
                    row=2, col=1
                )
            
            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        name='Sell',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Trading Strategy Backtest',
            height=800,
            showlegend=True
        )
        
        # Save chart if requested
        if save:
            self.save_plotly_chart(fig, f'{symbol}_backtest')
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, symbols: List[str], save: bool = True) -> go.Figure:
        """
        Create correlation heatmap for multiple stocks
        
        Args:
            df: DataFrame with returns for multiple stocks
            symbols: List of stock symbols
            save: Whether to save the chart
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            logger.error("Empty DataFrame for correlation analysis")
            return go.Figure()
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Stock Correlation Matrix',
            width=600,
            height=600
        )
        
        # Save chart if requested
        if save:
            self.save_plotly_chart(fig, 'correlation_heatmap')
        
        return fig
    
    def create_performance_comparison(self, results: Dict, save: bool = True) -> go.Figure:
        """
        Create performance comparison chart for multiple strategies
        
        Args:
            results: Dictionary with strategy results
            save: Whether to save the chart
            
        Returns:
            Plotly figure object
        """
        if not results:
            logger.error("No results provided for performance comparison")
            return go.Figure()
        
        # Extract metrics
        strategies = list(results.keys())
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate')
        )
        
        metric_info = [
            ('total_return', 'Total Return', 1, 1),
            ('sharpe_ratio', 'Sharpe Ratio', 1, 2),
            ('max_drawdown', 'Max Drawdown', 2, 1),
            ('win_rate', 'Win Rate', 2, 2)
        ]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for metric, title, row, col in metric_info:
            values = []
            strategy_names = []
            
            for strategy, result in results.items():
                if 'error' not in result and metric in result:
                    values.append(result[metric])
                    strategy_names.append(strategy)
            
            if values:
                fig.add_trace(
                    go.Bar(
                        x=strategy_names,
                        y=values,
                        name=title,
                        marker_color=colors[:len(values)]
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title='Strategy Performance Comparison',
            height=800,
            showlegend=False
        )
        
        # Save chart if requested
        if save:
            self.save_plotly_chart(fig, 'strategy_comparison')
        
        return fig
    
    def save_plotly_chart(self, fig: go.Figure, filename: str):
        """
        Save Plotly chart as HTML
        
        Args:
            fig: Plotly figure object
            filename: Base filename
        """
        try:
            filepath = os.path.join(self.output_dir, f"{filename}.html")
            pyo.plot(fig, filename=filepath, auto_open=False)
            logger.info(f"Chart saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")
    
    def save_matplotlib_chart(self, fig: plt.Figure, filename: str):
        """
        Save Matplotlib chart as PNG
        
        Args:
            fig: Matplotlib figure object
            filename: Base filename
        """
        try:
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Chart saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")
    
    def create_summary_dashboard(self, df: pd.DataFrame, symbol: str, save: bool = True) -> go.Figure:
        """
        Create comprehensive summary dashboard
        
        Args:
            df: DataFrame with all indicators
            symbol: Stock symbol
            save: Whether to save the chart
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            logger.error("Empty DataFrame for dashboard")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price & Moving Averages',
                'Volume',
                'RSI',
                'MACD',
                'Bollinger Bands',
                'Returns'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='blue')),
            row=1, col=1
        )
        
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='red')),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='green')),
                row=1, col=1
            )
        
        # Volume
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='lightblue'),
                row=1, col=2
            )
        
        # RSI
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi_14'], name='RSI 14', line=dict(color='purple')),
                row=2, col=1
            )
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')),
                row=2, col=2
            )
        
        if 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')),
                row=2, col=2
            )
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='red')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_middle'], name='BB Middle', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='green')),
                row=3, col=1
            )
        
        # Returns
        if 'returns' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['returns'], name='Returns', line=dict(color='orange')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Analysis Dashboard',
            height=1200,
            showlegend=True
        )
        
        # Save chart if requested
        if save:
            self.save_plotly_chart(fig, f'{symbol}_dashboard')
        
        return fig

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import StockDataFetcher
    from data_cleaner import StockDataCleaner
    from technical_indicators import TechnicalIndicators
    
    print("Testing visualizer...")
    
    # Fetch, clean, and calculate indicators
    fetcher = StockDataFetcher()
    raw_data = fetcher.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    
    if not raw_data.empty:
        cleaner = StockDataCleaner()
        clean_data, _ = cleaner.clean_data(raw_data, 'AAPL')
        
        calculator = TechnicalIndicators()
        indicators_data = calculator.calculate_all_indicators(clean_data)
        
        # Create visualizer
        visualizer = StockVisualizer()
        
        # Test different chart types
        print("Creating price chart...")
        price_chart = visualizer.create_price_chart(indicators_data, 'AAPL')
        
        print("Creating indicator chart...")
        indicator_chart = visualizer.create_indicator_chart(
            indicators_data, 
            ['rsi_14', 'macd', 'volume'], 
            'AAPL'
        )
        
        print("Creating returns analysis...")
        returns_chart = visualizer.create_returns_analysis(indicators_data, 'AAPL')
        
        print("Creating summary dashboard...")
        dashboard = visualizer.create_summary_dashboard(indicators_data, 'AAPL')
        
        print("Charts created successfully!")
    else:
        print("No data available for testing")
