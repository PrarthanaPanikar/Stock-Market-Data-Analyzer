"""
Streamlit Dashboard for Stock Market Data Analyzer
Interactive web interface for stock analysis
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import StockDataFetcher
from src.data_cleaner import StockDataCleaner
from src.technical_indicators import TechnicalIndicators
from src.backtest_engine import BacktestEngine
from src.visualizer import StockVisualizer
from src.report_generator import ReportGenerator

# Configure page
st.set_page_config(
    page_title="Stock Market Data Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
    .neutral {
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    """
    Streamlit dashboard class
    """
    
    def __init__(self):
        """Initialize dashboard"""
        self.fetcher = StockDataFetcher()
        self.cleaner = StockDataCleaner()
        self.indicators = TechnicalIndicators()
        self.backtest = BacktestEngine()
        self.visualizer = StockVisualizer()
        self.reporter = ReportGenerator()
        
        # Session state initialization
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'symbol' not in st.session_state:
            st.session_state.symbol = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
    
    def run(self):
        """Main dashboard runner"""
        st.markdown('<h1 class="main-header">📈 Stock Market Data Analyzer</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if st.session_state.symbol and st.session_state.data is not None:
            self.render_main_content()
        else:
            self.render_welcome()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("📊 Analysis Controls")
        
        # Stock selection
        symbol = st.sidebar.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Date range selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.sidebar.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Analysis start date"
            )
        with col2:
            end_date = st.sidebar.date_input(
                "End Date",
                value=datetime.now(),
                help="Analysis end date"
            )
        
        # Strategy selection
        st.sidebar.subheader("🎯 Trading Strategy")
        strategy = st.sidebar.selectbox(
            "Select Strategy",
            options=['sma_crossover', 'rsi_overbought_oversold', 'macd_crossover', 'bollinger_bands', 'combined'],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Choose trading strategy for backtesting"
        )
        
        # Backtest parameters
        st.sidebar.subheader("💰 Backtest Settings")
        initial_capital = st.sidebar.number_input(
            "Initial Capital ($)",
            value=100000,
            min_value=1000,
            max_value=10000000,
            step=1000,
            help="Initial capital for backtesting"
        )
        
        # Analysis button
        analyze_button = st.sidebar.button(
            "🚀 Run Analysis",
            type="primary",
            help="Click to start analysis"
        )
        
        # Run analysis when button is clicked
        if analyze_button:
            self.run_analysis(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), strategy, initial_capital)
        
        # Quick actions
        st.sidebar.subheader("⚡ Quick Actions")
        if st.sidebar.button("📥 Download Data"):
            self.download_data()
        
        if st.sidebar.button("📄 Generate Report"):
            self.generate_report()
        
        if st.sidebar.button("🔄 Reset"):
            self.reset_session()
    
    def run_analysis(self, symbol: str, start_date: str, end_date: str, strategy: str, initial_capital: float):
        """Run stock analysis"""
        try:
            with st.spinner("Fetching stock data..."):
                raw_data = self.fetcher.fetch_data(symbol, start_date, end_date)
                
                if raw_data.empty:
                    st.error(f"No data found for {symbol}")
                    return
            
            with st.spinner("Cleaning data..."):
                clean_data, cleaning_report = self.cleaner.clean_data(raw_data, symbol)
            
            with st.spinner("Calculating indicators..."):
                indicators_data = self.indicators.calculate_all_indicators(clean_data)
            
            with st.spinner("Generating signals..."):
                signals_data = self.indicators.generate_trading_signals(indicators_data, strategy)
            
            with st.spinner("Running backtest..."):
                self.backtest.initial_capital = initial_capital
                backtest_results = self.backtest.run_backtest(signals_data, 'signal')
            
            # Store results in session state
            st.session_state.symbol = symbol
            st.session_state.data = indicators_data
            st.session_state.analysis_results = {
                'symbol': symbol,
                'cleaning_report': cleaning_report,
                'indicators_summary': self.indicators.get_indicator_summary(indicators_data),
                'backtest_results': backtest_results,
                'strategy': strategy,
                'initial_capital': initial_capital
            }
            
            st.success(f"Analysis completed for {symbol}!")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    
    def render_main_content(self):
        """Render main content area"""
        if not st.session_state.data or not st.session_state.analysis_results:
            return
        
        symbol = st.session_state.symbol
        data = st.session_state.data
        results = st.session_state.analysis_results
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📈 Technical Analysis", "💰 Backtest", "📋 Data Quality", "📄 Reports"
        ])
        
        with tab1:
            self.render_overview_tab(symbol, data, results)
        
        with tab2:
            self.render_technical_analysis_tab(symbol, data, results)
        
        with tab3:
            self.render_backtest_tab(symbol, results)
        
        with tab4:
            self.render_data_quality_tab(results)
        
        with tab5:
            self.render_reports_tab(symbol, results)
    
    def render_overview_tab(self, symbol: str, data: pd.DataFrame, results: dict):
        """Render overview tab"""
        st.header(f"📊 {symbol} Stock Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = data['close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            price_change = data['close'].iloc[-1] - data['close'].iloc[0]
            price_change_pct = (price_change / data['close'].iloc[0]) * 100
            delta_color = "normal" if price_change_pct >= 0 else "inverse"
            st.metric("Price Change", f"{price_change_pct:.2f}%", f"${price_change:.2f}", delta_color)
        
        with col3:
            volatility = data['returns'].std() * np.sqrt(252) if 'returns' in data.columns else 0
            st.metric("Volatility", f"{volatility:.2%}")
        
        with col4:
            if 'backtest_results' in results and 'total_return' in results['backtest_results']:
                total_return = results['backtest_results']['total_return']
                st.metric("Strategy Return", f"{total_return:.2%}")
        
        # Price chart
        st.subheader("📈 Price Chart")
        fig = self.create_price_chart(data, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key statistics
        st.subheader("📊 Key Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Price Statistics**")
            stats_data = {
                'Period High': f"${data['high'].max():.2f}",
                'Period Low': f"${data['low'].min():.2f}",
                'Average Price': f"${data['close'].mean():.2f}",
                'Trading Days': len(data)
            }
            for key, value in stats_data.items():
                st.write(f"• {key}: {value}")
        
        with col2:
            st.write("**Volume Statistics**")
            if 'volume' in data.columns:
                volume_data = {
                    'Average Volume': f"{data['volume'].mean():,.0f}",
                    'Max Volume': f"{data['volume'].max():,.0f}",
                    'Min Volume': f"{data['volume'].min():,.0f}"
                }
                for key, value in volume_data.items():
                    st.write(f"• {key}: {value}")
    
    def render_technical_analysis_tab(self, symbol: str, data: pd.DataFrame, results: dict):
        """Render technical analysis tab"""
        st.header(f"📈 {symbol} Technical Analysis")
        
        # Indicator selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_ma = st.checkbox("Moving Averages", value=True)
            show_rsi = st.checkbox("RSI", value=True)
        
        with col2:
            show_macd = st.checkbox("MACD", value=True)
            show_bb = st.checkbox("Bollinger Bands", value=True)
        
        with col3:
            show_volume = st.checkbox("Volume", value=True)
            show_signals = st.checkbox("Trading Signals", value=True)
        
        # Create technical chart
        fig = self.create_technical_chart(data, symbol, {
            'show_ma': show_ma,
            'show_rsi': show_rsi,
            'show_macd': show_macd,
            'show_bb': show_bb,
            'show_volume': show_volume,
            'show_signals': show_signals
        })
        st.plotly_chart(fig, use_container_width=True)
        
        # Current indicators
        st.subheader("📊 Current Technical Indicators")
        if 'indicators_summary' in results:
            summary = results['indicators_summary']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Moving Averages**")
                if 'moving_averages' in summary:
                    for ma, value in summary['moving_averages']['latest_values'].items():
                        st.write(f"• {ma}: ${value:.2f}" if value else f"• {ma}: N/A")
            
            with col2:
                st.write("**Momentum**")
                if 'momentum' in summary:
                    for indicator, value in summary['momentum'].items():
                        st.write(f"• {indicator}: {value:.2f}" if value else f"• {indicator}: N/A")
            
            with col3:
                st.write("**Volatility**")
                if 'volatility' in summary:
                    for indicator, value in summary['volatility'].items():
                        if 'volatility' in indicator:
                            st.write(f"• {indicator}: {value:.2%}" if value else f"• {indicator}: N/A")
                        else:
                            st.write(f"• {indicator}: ${value:.2f}" if value else f"• {indicator}: N/A")
    
    def render_backtest_tab(self, symbol: str, results: dict):
        """Render backtest results tab"""
        st.header(f"💰 {symbol} Backtest Results")
        
        if 'backtest_results' not in results:
            st.warning("No backtest results available")
            return
        
        backtest_results = results['backtest_results']
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{backtest_results['total_return']:.2%}")
        
        with col2:
            st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.2%}")
        
        with col4:
            st.metric("Win Rate", f"{backtest_results['win_rate']:.2%}")
        
        # Detailed metrics
        st.subheader("📊 Detailed Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Returns & Risk**")
            metrics_data = {
                'Initial Capital': f"${backtest_results['initial_capital']:,.2f}",
                'Final Value': f"${backtest_results['final_value']:,.2f}",
                'Annualized Return': f"{backtest_results.get('annualized_return', 0):.2%}",
                'Volatility': f"{backtest_results.get('volatility', 0):.2%}"
            }
            for key, value in metrics_data.items():
                st.write(f"• {key}: {value}")
        
        with col2:
            st.write("**Trading Statistics**")
            trading_data = {
                'Total Trades': backtest_results['total_trades'],
                'Winning Trades': backtest_results['winning_trades'],
                'Losing Trades': backtest_results['losing_trades'],
                'Profit Factor': f"{backtest_results.get('profit_factor', 0):.2f}",
                'Average Trade': f"${backtest_results.get('average_trade', 0):.2f}"
            }
            for key, value in trading_data.items():
                st.write(f"• {key}: {value}")
        
        # Equity curve
        if 'equity_curve' in backtest_results:
            st.subheader("📈 Equity Curve")
            equity_fig = self.create_equity_curve_chart(backtest_results['equity_curve'], backtest_results.get('trades', []))
            st.plotly_chart(equity_fig, use_container_width=True)
    
    def render_data_quality_tab(self, results: dict):
        """Render data quality tab"""
        st.header("📋 Data Quality Assessment")
        
        if 'cleaning_report' not in results:
            st.warning("No data quality information available")
            return
        
        cleaning_report = results['cleaning_report']
        
        # Quality score
        quality_score = cleaning_report.get('data_quality_score', 0)
        st.metric("Data Quality Score", f"{quality_score:.2f}/100")
        
        # Cleaning steps
        st.subheader("🔧 Data Processing Steps")
        for step in cleaning_report.get('cleaning_steps', []):
            step_name = step.get('step', 'Unknown step')
            with st.expander(f"Step: {step_name.replace('_', ' ').title()}"):
                if 'duplicate_count' in step:
                    st.write(f"Duplicates removed: {step['duplicate_count']}")
                
                if 'missing_info' in step and step['missing_info']:
                    st.write("Missing values found:")
                    for col, count in step['missing_info'].items():
                        st.write(f"  • {col}: {count}")
                
                if 'strategies_used' in step:
                    st.write("Cleaning strategies applied:")
                    for strategy in step['strategies_used']:
                        st.write(f"  • {strategy}")
                
                if 'outliers_detected' in step and step['outliers_detected']:
                    st.write("Outliers detected:")
                    for indicator, count in step['outliers_detected'].items():
                        st.write(f"  • {indicator}: {count}")
                
                if 'corrections' in step and step['corrections']:
                    st.write("Corrections made:")
                    for correction in step['corrections'].values():
                        st.write(f"  • {correction}")
        
        # Data summary
        st.subheader("📊 Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data**")
            st.write(f"• Shape: {cleaning_report['original_shape']}")
        
        with col2:
            st.write("**Cleaned Data**")
            st.write(f"• Shape: {cleaning_report['final_shape']}")
            st.write(f"• Rows removed: {cleaning_report['original_shape'][0] - cleaning_report['final_shape'][0]}")
    
    def render_reports_tab(self, symbol: str, results: dict):
        """Render reports tab"""
        st.header(f"📄 {symbol} Analysis Reports")
        
        # Generate report button
        if st.button("📊 Generate Full Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                try:
                    report_path = self.reporter.generate_comprehensive_report(
                        st.session_state.data,
                        symbol,
                        results.get('backtest_results'),
                        results.get('indicators_summary')
                    )
                    st.success(f"Report generated successfully!")
                    st.info(f"Report saved to: {report_path}")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Summary statistics
        st.subheader("📋 Analysis Summary")
        
        if 'indicators_summary' in results:
            summary = results['indicators_summary']
            
            # Strategy summary
            st.write("**Strategy Information**")
            st.write(f"• Strategy: {results.get('strategy', 'N/A')}")
            st.write(f"• Initial Capital: ${results.get('initial_capital', 0):,.2f}")
            
            if 'backtest_results' in results:
                backtest = results['backtest_results']
                st.write(f"• Total Return: {backtest.get('total_return', 0):.2%}")
                st.write(f"• Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.2f}")
                st.write(f"• Win Rate: {backtest.get('win_rate', 0):.2%}")
        
        # Download options
        st.subheader("💾 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download CSV"):
                self.download_data()
        
        with col2:
            if st.button("📈 Download Chart"):
                self.download_chart()
        
        with col3:
            if st.button("📄 Download Summary"):
                self.download_summary()
    
    def create_price_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create price chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price', 'Volume'),
            row_width=[0.2, 0.7]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['sma_20'], name='SMA 20', line=dict(color='red')),
                row=1, col=1
            )
        
        if 'sma_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['sma_50'], name='SMA 50', line=dict(color='green')),
                row=1, col=1
            )
        
        # Volume
        if 'volume' in data.columns:
            fig.add_trace(
                go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        return fig
    
    def create_technical_chart(self, data: pd.DataFrame, symbol: str, options: dict) -> go.Figure:
        """Create comprehensive technical chart"""
        # Determine subplot layout
        subplot_count = 1
        if options.get('show_rsi') or options.get('show_macd'):
            subplot_count += 1
        if options.get('show_volume'):
            subplot_count += 1
        
        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Price & Indicators', 'Momentum', 'Volume'][:subplot_count]
        )
        
        row = 1
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['close'], name='Close', line=dict(color='blue')),
            row=row, col=1
        )
        
        if options.get('show_ma'):
            if 'sma_20' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['sma_20'], name='SMA 20', line=dict(color='red')), row=row, col=1)
            if 'sma_50' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], name='SMA 50', line=dict(color='green')), row=row, col=1)
        
        if options.get('show_bb'):
            if all(col in data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], name='BB Upper', line=dict(color='red')), row=row, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['bb_middle'], name='BB Middle', line=dict(color='blue')), row=row, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], name='BB Lower', line=dict(color='green')), row=row, col=1)
        
        # Trading signals
        if options.get('show_signals') and 'signal' in data.columns:
            buy_signals = data[data['signal'] == 1]
            sell_signals = data[data['signal'] == -1]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', 
                              name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')),
                    row=row, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers',
                              name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')),
                    row=row, col=1
                )
        
        # Momentum indicators
        if options.get('show_rsi') or options.get('show_macd'):
            row += 1
            
            if options.get('show_rsi') and 'rsi_14' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['rsi_14'], name='RSI 14', line=dict(color='purple')),
                    row=row, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
            
            if options.get('show_macd'):
                if 'macd' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['macd'], name='MACD', line=dict(color='blue')),
                        row=row, col=1
                    )
                if 'macd_signal' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['macd_signal'], name='Signal', line=dict(color='red')),
                        row=row, col=1
                    )
        
        # Volume
        if options.get('show_volume') and 'volume' in data.columns:
            row += 1
            fig.add_trace(
                go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='lightblue'),
                row=row, col=1
            )
        
        fig.update_layout(height=300 * subplot_count, showlegend=True)
        return fig
    
    def create_equity_curve_chart(self, equity_curve: pd.DataFrame, trades: list) -> go.Figure:
        """Create equity curve chart"""
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            )
        )
        
        # Trade markers
        if trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates, y=buy_prices, mode='markers',
                        name='Buy', marker=dict(color='green', size=8, symbol='triangle-up')
                    )
                )
            
            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates, y=sell_prices, mode='markers',
                        name='Sell', marker=dict(color='red', size=8, symbol='triangle-down')
                    )
                )
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def download_data(self):
        """Download data as CSV"""
        if st.session_state.data is not None:
            csv = st.session_state.data.to_csv(index=True)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{st.session_state.symbol}_data.csv",
                mime="text/csv"
            )
    
    def download_chart(self):
        """Download chart as image"""
        st.info("Chart download feature coming soon!")
    
    def download_summary(self):
        """Download summary as text"""
        if st.session_state.analysis_results:
            summary = self.generate_text_summary(st.session_state.analysis_results)
            st.download_button(
                label="📄 Download Summary",
                data=summary,
                file_name=f"{st.session_state.symbol}_summary.txt",
                mime="text/plain"
            )
    
    def generate_text_summary(self, results: dict) -> str:
        """Generate text summary"""
        lines = []
        lines.append(f"Stock Analysis Summary - {results['symbol']}")
        lines.append("=" * 50)
        
        if 'backtest_results' in results:
            bt = results['backtest_results']
            lines.append(f"Strategy: {results['strategy']}")
            lines.append(f"Initial Capital: ${results['initial_capital']:,.2f}")
            lines.append(f"Total Return: {bt.get('total_return', 0):.2%}")
            lines.append(f"Sharpe Ratio: {bt.get('sharpe_ratio', 0):.2f}")
            lines.append(f"Win Rate: {bt.get('win_rate', 0):.2%}")
            lines.append(f"Total Trades: {bt.get('total_trades', 0)}")
        
        return "\n".join(lines)
    
    def generate_report(self):
        """Generate comprehensive report"""
        if st.session_state.data and st.session_state.analysis_results:
            try:
                report_path = self.reporter.generate_comprehensive_report(
                    st.session_state.data,
                    st.session_state.symbol,
                    st.session_state.analysis_results.get('backtest_results'),
                    st.session_state.analysis_results.get('indicators_summary')
                )
                st.success(f"Report generated: {report_path}")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    def reset_session(self):
        """Reset session state"""
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()
    
    def render_welcome(self):
        """Render welcome screen"""
        st.markdown("""
        ## 🎯 Welcome to Stock Market Data Analyzer!
        
        This comprehensive tool provides:
        
        ### 📊 **Features**
        - **Real-time Data Analysis**: Fetch stock data from Yahoo Finance
        - **Technical Indicators**: 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - **Strategy Backtesting**: Test multiple trading strategies
        - **Interactive Charts**: Beautiful, interactive visualizations
        - **Risk Analysis**: Comprehensive risk metrics and VaR calculations
        - **Report Generation**: Professional PDF reports
        
        ### 🚀 **Getting Started**
        1. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
        2. Select date range for analysis
        3. Choose trading strategy
        4. Click "Run Analysis" to start
        
        ### 📈 **Supported Strategies**
        - **SMA Crossover**: Simple moving average crossover strategy
        - **RSI Overbought/Oversold**: RSI-based mean reversion
        - **MACD Crossover**: MACD signal line crossover
        - **Bollinger Bands**: Price reversal at bands
        - **Combined**: Multi-indicator consensus approach
        
        ---
        
        **Enter a stock symbol in the sidebar to get started!** 📊
        """)

def main():
    """Main function"""
    dashboard = StreamlitDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
