"""
Stock Prediction Platform - Main Application
A modular stock prediction platform using Streamlit that fetches real-time data 
and allows incremental algorithm additions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import our custom modules
from modules.data_fetcher import StockDataFetcher
from modules.algorithms import PredictionAlgorithms
from modules.predictor import StockPredictor
from modules.utils import ChartUtils, DataUtils, ValidationUtils

def main():
    """
    Main application function.
    This is the entry point of our stock prediction platform.
    """
    
    # Configure page
    st.set_page_config(
        page_title="Stock Prediction Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StockPredictor()
    

    
    # Top navigation bar
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("📈 Stock Prediction Platform")
        st.markdown("### A Beginner-Friendly Modular Stock Prediction System")
    
    with col2:
        # Sidebar reminder
        st.markdown("**📊 Use sidebar →**")
        st.caption("Stock selection controls")
    
    with col3:
        # Cache clearing for debugging
        if st.button("🔄 Clear Cache", help="Clear data cache for fresh data"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Stock Analysis", 
        "📊 Predictions", 
        "📈 Advanced Analysis", 
        "ℹ️ About & Help"
    ])
    
    with tab1:
        stock_analysis_tab()
    
    with tab2:
        predictions_tab()
    
    with tab3:
        advanced_analysis_tab()
    
    with tab4:
        about_help_tab()

def stock_analysis_tab():
    """
    Tab for basic stock analysis and data exploration.
    This is where beginners can start understanding stock data.
    """
    
    st.header("🔍 Stock Data Analysis")
    st.markdown("**Start here to understand stock data and see basic visualizations.**")
    
    # Sidebar for stock selection - Always visible
    with st.sidebar:
        st.header("📊 Stock Selection")
        st.markdown("**🔹 Select stocks here**")
        
        # Market selection - Indian Market Only
        market_choice = "Indian Market"
        st.write("**Market:** Indian Stock Market (NSE)")
        st.write("**Currency:** Indian Rupees (₹)")
        
        # Popular stocks dropdown based on market
        popular_stocks = st.session_state.predictor.data_fetcher.get_market_stocks(market_choice)
        
        # Stock selection - simplified to dropdown only
        stock_symbol = st.selectbox(
            "Select Stock:",
            [""] + popular_stocks,
            help="Choose from popular Indian stocks"
        )
        
        # Time period selection
        period = st.selectbox(
            "Time Period:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Select how much historical data to analyze"
        )
        
        # Analyze button
        analyze_button = st.button("📈 Analyze Stock", type="primary")
    
    # Main content area
    if stock_symbol and analyze_button:
        # Validate symbol
        is_valid, error_msg = ValidationUtils.validate_stock_symbol(stock_symbol)
        
        if not is_valid:
            st.error(error_msg)
            return
        
        # Show loading spinner
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            # Get stock summary
            stock_summary = st.session_state.predictor.get_stock_summary(stock_symbol)
            
            if 'error' in stock_summary:
                st.error(stock_summary['error'])
                return
            
            # Get historical data
            historical_data = st.session_state.predictor.data_fetcher.fetch_stock_data(stock_symbol, period)
            
            if historical_data is None:
                st.error("Failed to fetch historical data")
                return
        
        # Navigation and stock information header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"📊 {stock_symbol} - Stock Information")
        with col2:
            if st.button("← Back to Menu", type="secondary"):
                st.rerun()
        
        # Create columns for stock info
        col1, col2, col3, col4 = st.columns(4)
        
        # Detect currency type for proper formatting
        currency_type = DataUtils.detect_currency_from_symbol(stock_symbol)


        
        with col1:
            st.metric(
                "Current Price",
                DataUtils.format_currency(stock_summary.get('current_price'), currency_type),
                delta=DataUtils.format_currency(stock_summary.get('price_change'), currency_type)
            )
        
        with col2:
            if stock_summary.get('info'):
                st.metric(
                    "Market Cap",
                    DataUtils.format_number(stock_summary['info'].get('market_cap', 'N/A'))
                )
        
        with col3:
            st.metric(
                "Volume",
                DataUtils.format_number(stock_summary.get('volume'))
            )
        
        with col4:
            trend = DataUtils.get_trend_indicator(stock_summary.get('price_change'))
            st.metric(
                "Trend",
                trend
            )
        
        # Company information
        if stock_summary.get('info'):
            info = stock_summary['info']
            
            st.subheader("🏢 Company Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Company:** {info.get('name', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            
            with col2:
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        
        # Price chart
        st.subheader("📈 Price Chart")
        
        price_chart = ChartUtils.create_price_chart(
            historical_data,
            title=f"{stock_symbol} Stock Price - {period.upper()}"
        )
        
        if price_chart:
            st.plotly_chart(price_chart, use_container_width=True)
        
        # Data table
        st.subheader("📋 Recent Data")
        
        # Show last 10 days of data
        recent_data = historical_data.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        recent_data['Date'] = pd.to_datetime(recent_data['Date']).dt.strftime('%Y-%m-%d')
        
        # Format numbers with proper currency
        for col in ['Open', 'High', 'Low', 'Close']:
            recent_data[col] = recent_data[col].apply(lambda x: DataUtils.format_currency(x, currency_type))
        
        recent_data['Volume'] = recent_data['Volume'].apply(DataUtils.format_number)
        
        st.dataframe(recent_data, use_container_width=True, hide_index=True)
        
        # Store data in session state for use in other tabs
        st.session_state.current_stock = stock_symbol
        st.session_state.current_data = historical_data
        st.session_state.current_period = period
        
        # Navigation tips
        st.markdown("---")
        st.info("💡 **Next Step:** Go to the 'Predictions' tab above to generate price predictions for this stock!")
        
        # Quick navigation reminder
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("📊 **Analysis Complete!** You can now:")
            st.markdown("• Switch to **Predictions** tab to forecast future prices")
            st.markdown("• Select a different stock from the sidebar")
            st.markdown("• Change the time period and re-analyze")
        with col2:
            if st.button("🔄 Select New Stock", key="new_stock"):
                st.rerun()
    
    else:
        # Show instructions
        # Show sidebar instruction with button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("👆 Select a stock symbol from the sidebar and click 'Analyze Stock' to get started!")
        with col2:
            st.markdown("**👉 Use Sidebar**")
            st.caption("Select stock from the panel on the right →")
        
        st.markdown("""
        ### 🎯 What This Section Does:
        
        1. **Stock Selection**: Choose from popular Indian stocks or enter any stock symbol
        2. **Time Period**: Select how much historical data to analyze
        3. **Company Info**: See basic company information and current metrics in ₹
        4. **Price Chart**: Interactive chart showing price movements
        5. **Recent Data**: Table showing recent trading data
        
        ### 📚 For Beginners:
        
        - **Stock Symbol**: A unique identifier for Indian companies
          - Examples: RELIANCE.NS (Reliance), TCS.NS (TCS), HDFCBANK.NS (HDFC Bank)
        - **Open/Close**: Opening and closing prices for each trading day in ₹
        - **High/Low**: Highest and lowest prices during the day in ₹
        - **Volume**: Number of shares traded
        - **Market Cap**: Total value of all company shares
        - **.NS suffix**: Required for Indian stocks on NSE (National Stock Exchange)
        - **Currency**: All prices shown in Indian Rupees (₹)
        """)

def predictions_tab():
    """
    Tab for making stock price predictions.
    This is where the magic happens - algorithms predict future prices.
    """
    
    # Navigation and predictions header  
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("📊 Stock Price Predictions")
        st.markdown("**Use algorithms to predict future stock prices.**")
    with col2:
        if st.button("← Back to Analysis", key="pred_back", type="secondary"):
            st.session_state.pop('current_stock', None)
            st.session_state.pop('current_data', None)
            st.rerun()
    
    # Check if we have current stock data
    if 'current_stock' not in st.session_state:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning("Please analyze a stock in the 'Stock Analysis' tab first!")
        with col2:
            st.markdown("**👉 Use Sidebar**")
            st.caption("Go to 'Stock Analysis' tab and select a stock from the sidebar →")
        return
    
    stock_symbol = st.session_state.current_stock
    historical_data = st.session_state.current_data
    
    st.subheader(f"🔮 Predictions for {stock_symbol}")
    
    # Prediction parameters
    with st.sidebar:
        st.header("🎛️ Prediction Settings")
        
        # Algorithm selection
        available_algorithms = st.session_state.predictor.algorithms.get_available_algorithms()
        
        selected_algorithms = st.multiselect(
            "Select Algorithms:",
            available_algorithms,
            default=available_algorithms[:1],  # Select first algorithm by default
            help="Choose which algorithms to use for prediction"
        )
        
        # Algorithm-specific parameters
        st.subheader("⚙️ Algorithm Parameters")
        
        # Common parameter for all algorithms
        predict_days = st.slider(
            "Days to Predict:",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of future days to predict (up to 1 year)"
        )
        
        # Algorithm-specific parameters
        algorithm_params = {}
        
        if 'Simple Moving Average' in selected_algorithms:
            st.write("**Simple Moving Average Settings:**")
            sma_window = st.slider(
                "Moving Average Window:",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of days to calculate moving average",
                key="sma_window"
            )
            algorithm_params['sma_window'] = sma_window
        
        if 'Linear Regression' in selected_algorithms:
            st.write("**Linear Regression Settings:**")
            lr_window = st.slider(
                "Feature Window:",
                min_value=10,
                max_value=60,
                value=20,
                help="Number of days for feature calculation",
                key="lr_window"
            )
            algorithm_params['lr_window'] = lr_window
        
        if 'Exponential Smoothing' in selected_algorithms:
            st.write("**Exponential Smoothing Settings:**")
            alpha = st.slider(
                "Smoothing Factor (Alpha):",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.1,
                help="Higher values give more weight to recent data",
                key="es_alpha"
            )
            algorithm_params['alpha'] = alpha
        
        if 'LSTM Neural Network' in selected_algorithms:
            st.write("**LSTM Settings:**")
            sequence_length = st.slider(
                "Sequence Length:",
                min_value=30,
                max_value=100,
                value=60,
                help="Number of days the AI model looks back",
                key="lstm_seq"
            )
            algorithm_params['sequence_length'] = sequence_length
        
        if 'Advanced Ensemble' in selected_algorithms:
            st.write("**Advanced Ensemble Settings:**")
            st.info("🎯 This algorithm uses RSI, MACD, Bollinger Bands, and multiple timeframe analysis for maximum accuracy!")
            # No additional parameters needed for ensemble
        
        if 'ARIMA Model' in selected_algorithms:
            st.write("**ARIMA Model Settings:**")
            st.info("📊 Professional time series model used by financial institutions")
            arima_col1, arima_col2, arima_col3 = st.columns(3)
            with arima_col1:
                p = st.slider("AR Order (p):", 1, 5, 2, key="arima_p")
            with arima_col2:
                d = st.slider("Diff Order (d):", 0, 2, 1, key="arima_d")
            with arima_col3:
                q = st.slider("MA Order (q):", 1, 5, 2, key="arima_q")
            algorithm_params['arima_order'] = (p, d, q)
        
        if 'Prophet Model' in selected_algorithms:
            st.write("**Prophet Model Settings:**")
            st.info("🔮 Facebook's advanced forecasting with seasonality detection")
            # Prophet handles parameters automatically
        
        if 'Random Forest' in selected_algorithms:
            st.write("**Random Forest Settings:**")
            st.info("🌲 Ensemble ML used by quantitative hedge funds")
            n_trees = st.slider("Number of Trees:", 50, 200, 100, step=10, key="rf_trees")
            algorithm_params['n_estimators'] = n_trees
        
        if 'XGBoost Model' in selected_algorithms:
            st.write("**XGBoost Settings:**")
            st.info("🚀 Gradient boosting - industry standard for ML competitions")
            # XGBoost uses optimized default parameters
        
        if 'Kalman Filter' in selected_algorithms:
            st.write("**Kalman Filter Settings:**")
            st.info("🎯 State space model used by institutional traders")
            # Kalman Filter uses optimized parameters
        
        # Prediction button
        predict_button = st.button("🔮 Generate Predictions", type="primary")
    
    # Generate predictions
    if predict_button and selected_algorithms:
        # Validate parameters
        window_param = algorithm_params.get('sma_window', 20)  # Default for validation
        is_valid, error_msg = ValidationUtils.validate_prediction_parameters(window_param, predict_days)
        
        if not is_valid:
            st.error(error_msg)
            return
        
        with st.spinner("Generating predictions..."):
            # Get predictions from selected algorithms with proper parameters
            prediction_results = st.session_state.predictor.get_multiple_predictions(
                stock_symbol,
                selected_algorithms,
                period=st.session_state.current_period,
                predict_days=predict_days,
                algorithm_params=algorithm_params
            )
        
        if 'error' in prediction_results:
            st.error(prediction_results['error'])
            return
        
        # Display results
        combined_predictions = prediction_results['combined_predictions']
        individual_predictions = prediction_results['individual_predictions']
        
        # Prediction chart
        st.subheader("📈 Prediction Chart")
        
        try:
            predictions_data = combined_predictions.get('predictions', [])
            algorithm_data = combined_predictions.get('algorithm_data', None)
            
            # Use session state data if algorithm_data is not available
            if algorithm_data is None or algorithm_data.empty:
                algorithm_data = st.session_state.current_data
            
            # Validate data structure
            if algorithm_data is not None and not algorithm_data.empty:
                # Ensure Date column exists
                if 'Date' not in algorithm_data.columns:
                    algorithm_data = algorithm_data.reset_index()
                
                prediction_chart = ChartUtils.create_price_chart(
                    algorithm_data,
                    predictions_data,
                    title=f"{stock_symbol} Price Predictions"
                )
                
                if prediction_chart:
                    st.plotly_chart(prediction_chart, use_container_width=True)
                else:
                    st.error("Could not create prediction chart")
            else:
                st.warning("No historical data available for charting")
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")
            # Show debug info for troubleshooting
            with st.expander("Debug Info"):
                st.write(f"Predictions data type: {type(predictions_data)}")
                st.write(f"Algorithm data type: {type(algorithm_data)}")
                if predictions_data:
                    st.write(f"First prediction: {predictions_data[0] if len(predictions_data) > 0 else 'None'}")
                if algorithm_data is not None:
                    st.write(f"Algorithm data columns: {list(algorithm_data.columns) if hasattr(algorithm_data, 'columns') else 'No columns'}")
                    st.write(f"Algorithm data shape: {algorithm_data.shape if hasattr(algorithm_data, 'shape') else 'No shape'}")
                st.write(f"Session state data available: {hasattr(st.session_state, 'current_data') and st.session_state.current_data is not None}")
        
        # Predictions table
        st.subheader("📋 Predicted Prices")
        
        if combined_predictions.get('predictions'):
            pred_df = pd.DataFrame(combined_predictions['predictions'])
            pred_df['Date'] = pd.to_datetime(pred_df['Date']).dt.strftime('%Y-%m-%d')
            
            # Use proper currency formatting for predictions
            currency_type = DataUtils.detect_currency_from_symbol(stock_symbol)

            pred_df['Predicted_Price'] = pred_df['Predicted_Price'].apply(lambda x: DataUtils.format_currency(x, currency_type))
            
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Accuracy metrics
        st.subheader("📊 Accuracy Metrics")
        
        for algo_name, result in individual_predictions.items():
            st.write(f"**{algo_name}**")
            
            if 'metrics' in result:
                metrics = result['metrics']
                
                if 'error' not in metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Accuracy Score",
                            f"{metrics.get('Accuracy_Score', 0):.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "MAPE",
                            f"{metrics.get('MAPE', 0):.2f}%"
                        )
                    
                    with col3:
                        # Ensure currency detection is also done in predictions tab
                        pred_currency_type = DataUtils.detect_currency_from_symbol(stock_symbol)
                        st.metric(
                            "MAE",
                            DataUtils.format_currency(metrics.get('MAE', 0), pred_currency_type)
                        )
                    
                    with col4:
                        st.metric(
                            "R²",
                            f"{metrics.get('R_squared', 0):.3f}"
                        )
                    
                    # Confidence level
                    confidence = st.session_state.predictor.get_prediction_confidence(result)
                    
                    confidence_color = {
                        'High': 'green',
                        'Medium': 'orange',
                        'Low': 'red',
                        'Very Low': 'red'
                    }.get(confidence.get('confidence', 'Unknown'), 'gray')
                    
                    st.markdown(f"**Confidence Level:** :{confidence_color}[{confidence.get('confidence', 'Unknown')}] ({confidence.get('level', 0):.1f}%)")
                else:
                    st.error(f"Error calculating metrics: {metrics['error']}")
            
            st.markdown("---")
        
        # Explanation for beginners
        with st.expander("🤔 What do these metrics mean?"):
            st.markdown("""
            ### 📊 Accuracy Metrics Explained:
            
            - **Accuracy Score**: Overall accuracy percentage (higher is better)
            - **MAPE** (Mean Absolute Percentage Error): Average percentage error (lower is better)
            - **MAE** (Mean Absolute Error): Average dollar error (lower is better)
            - **R²** (R-squared): How well the model fits the data (closer to 1 is better)
            
            ### 🎯 Confidence Levels:
            
            - **High**: Predictions are likely to be reliable
            - **Medium**: Predictions have moderate reliability
            - **Low**: Predictions should be used with caution
            - **Very Low**: Predictions are not reliable
            
            ### ⚠️ Important Notes:
            
            - Past performance doesn't guarantee future results
            - Stock markets are inherently unpredictable
            - Use predictions as one factor among many in decision-making
            - Always do your own research before making investment decisions
            """)
    
    elif predict_button and not selected_algorithms:
        st.error("Please select at least one algorithm!")
    
    else:
        st.info("👆 Configure prediction settings in the sidebar and click 'Generate Predictions'!")
        
        st.markdown("""
        ### 🎯 How Predictions Work:
        
        1. **Algorithm Selection**: Choose which prediction algorithms to use
        2. **Parameter Configuration**: Set algorithm-specific parameters
        3. **Prediction Generation**: Algorithms analyze historical data and predict future prices
        4. **Results Display**: See predictions, charts, and accuracy metrics
        
        ### 📚 Current Algorithm: Simple Moving Average (SMA)
        
        **What it does:**
        - Calculates average price over a specified number of days
        - Smooths out price fluctuations to show trends
        - Predicts future prices based on recent trends
        
        **Why start with SMA:**
        - Easy to understand and implement
        - Good baseline for comparison with other algorithms
        - Commonly used in technical analysis
        
        **Parameters:**
        - **Window**: Number of days to calculate average (e.g., 20-day moving average)
        - **Predict Days**: How many days into the future to predict
        
        ### 🔮 Future Algorithms (Coming Soon):
        
        As you build your platform, you can add:
        - **Linear Regression**: Statistical trend analysis
        - **LSTM Neural Networks**: Deep learning for time series
        - **Prophet**: Facebook's time series forecasting
        - **ARIMA**: Statistical time series modeling
        - **Random Forest**: Ensemble machine learning
        """)

def advanced_analysis_tab():
    """
    Tab for advanced analysis and algorithm comparison.
    This is where users can dive deeper into the predictions.
    """
    
    st.header("📈 Advanced Analysis")
    st.markdown("**Deep dive into prediction algorithms and their performance.**")
    
    # Check if we have predictions
    if 'current_stock' not in st.session_state:
        st.warning("Please analyze a stock and generate predictions first!")
        return
    
    stock_symbol = st.session_state.current_stock
    
    st.subheader(f"🔬 Advanced Analysis for {stock_symbol}")
    
    # Algorithm comparison section
    st.subheader("⚖️ Algorithm Comparison")
    
    st.markdown("""
    This section will be enhanced as you add more algorithms to your platform.
    
    **Future Features:**
    - Side-by-side algorithm performance comparison
    - Backtesting results over different time periods
    - Risk-adjusted returns analysis
    - Ensemble method optimization
    """)
    
    # Model performance over time
    st.subheader("📊 Model Performance Analysis")
    
    # Placeholder for future advanced features
    st.info("🚧 This section will be populated as you add more algorithms and features.")
    
    # Algorithm insights
    with st.expander("🧠 Algorithm Insights"):
        st.markdown("""
        ### 🔍 Understanding Algorithm Behavior:
        
        **Simple Moving Average (SMA):**
        - **Strengths**: Simple, reliable for trend identification
        - **Weaknesses**: Lags behind price changes, poor in volatile markets
        - **Best For**: Stable stocks with clear trends
        
        **Tips for Improvement:**
        1. Combine with other indicators
        2. Use different timeframes
        3. Consider market conditions
        4. Add volume analysis
        """)
    
    # Future algorithm framework
    st.subheader("🏗️ Algorithm Development Framework")
    
    st.markdown("""
    ### 🎯 Your Modular Design Benefits:
    
    1. **Easy Algorithm Addition**: Each algorithm is a separate method
    2. **Consistent Interface**: All algorithms follow the same pattern
    3. **Automatic Combination**: New algorithms are automatically included in ensemble methods
    4. **Performance Tracking**: Built-in metrics for all algorithms
    
    ### 📋 Adding New Algorithms:
    
    To add a new algorithm to your platform:
    
    1. **Create Algorithm Method**: Add to `modules/algorithms.py`
    2. **Follow Interface**: Return predictions and metrics in standard format
    3. **Update Registry**: Add to the algorithms dictionary
    4. **Test Integration**: Algorithm automatically appears in UI
    
    ### 🔮 Recommended Next Algorithms:
    
    1. **Exponential Moving Average (EMA)**: More responsive than SMA
    2. **Linear Regression**: Statistical trend analysis
    3. **Bollinger Bands**: Volatility-based predictions
    4. **RSI-based Signals**: Momentum indicators
    """)

def about_help_tab():
    """
    Tab with information about the platform and help for beginners.
    """
    
    st.header("ℹ️ About Stock Prediction Platform")
    
    st.markdown("""
    ### 🎯 Purpose
    
    This platform is designed to help beginners learn about stock prediction while providing
    a solid foundation for adding more sophisticated algorithms over time.
    
    ### 🏗️ Modular Architecture
    
    The platform is built with modularity in mind:
    
    - **Data Fetcher**: Handles all data retrieval from Yahoo Finance
    - **Algorithms**: Contains prediction algorithms (easily expandable)
    - **Predictor**: Orchestrates the prediction process
    - **Utils**: Utility functions for charts and data processing
    - **Main App**: User interface and navigation
    
    ### 📚 Learning Path for Beginners
    
    1. **Start with Stock Analysis**: Understand basic stock data and charts
    2. **Try Simple Predictions**: Use the Simple Moving Average algorithm
    3. **Understand Metrics**: Learn what accuracy metrics mean
    4. **Experiment with Parameters**: See how different settings affect predictions
    5. **Add More Algorithms**: Gradually expand your toolkit
    
    ### 🔮 Future Enhancements
    
    Your platform is designed to grow. Here's what you can add:
    
    #### 📊 More Algorithms:
    - Exponential Moving Average (EMA)
    - Linear Regression
    - LSTM Neural Networks
    - Random Forest
    - Prophet (Facebook's forecasting)
    - ARIMA models
    
    #### 🎯 Advanced Features:
    - Real-time data updates
    - Portfolio optimization
    - Risk management tools
    - Backtesting framework
    - Paper trading simulation
    - Alert system
    
    #### 📱 UI Improvements:
    - Custom themes
    - Mobile optimization
    - Dashboard customization
    - Export capabilities
    
    ### ⚠️ Important Disclaimers
    
    - This is for educational purposes only
    - Past performance doesn't guarantee future results
    - Stock markets are inherently unpredictable
    - Always do your own research
    - Consider consulting with financial advisors
    - Never invest more than you can afford to lose
    """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### 📦 Technology Stack:
        
        - **Frontend**: Streamlit
        - **Data Source**: Yahoo Finance (via yfinance)
        - **Charts**: Plotly
        - **Data Processing**: Pandas, NumPy
        - **Future ML**: Scikit-learn ready
        
        ### 📁 File Structure:
        ```
        /
        ├── app.py (Main application)
        ├── modules/
        │   ├── data_fetcher.py (Data retrieval)
        │   ├── algorithms.py (Prediction algorithms)
        │   ├── predictor.py (Prediction orchestration)
        │   └── utils.py (Utility functions)
        └── .streamlit/
            └── config.toml (Streamlit configuration)
        ```
        
        ### 🎨 Design Principles:
        
        1. **Modularity**: Each component has a specific responsibility
        2. **Extensibility**: Easy to add new algorithms and features
        3. **Beginner-Friendly**: Clear explanations and help text
        4. **Professional**: Production-ready code structure
        5. **Maintainable**: Clean code with proper documentation
        """)
    
    # FAQ section
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        ### 🤔 Common Questions:
        
        **Q: How accurate are the predictions?**
        A: Accuracy varies by market conditions, stock volatility, and algorithm used. Always check the confidence levels and metrics.
        
        **Q: Can I use this for real trading?**
        A: This is educational software. Use it to learn, but always do additional research before making investment decisions.
        
        **Q: How often should I update predictions?**
        A: Daily updates are recommended for active trading, weekly for long-term analysis.
        
        **Q: What stocks work best with these algorithms?**
        A: Large-cap stocks with good trading volume tend to be more predictable than small-cap or highly volatile stocks.
        
        **Q: How do I add new algorithms?**
        A: Add new methods to the `PredictionAlgorithms` class in `modules/algorithms.py` following the existing pattern.
        
        **Q: Can I export the predictions?**
        A: Currently not implemented, but you can add export functionality to the platform.
        """)

# Run the application
if __name__ == "__main__":
    main()
