"""
Utility Module
This module contains helper functions and utilities used across the application.
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ChartUtils:
    """
    Utility class for creating charts and visualizations.
    
    Why separate chart utilities?
    - Reusable across different parts of the app
    - Consistent styling and formatting
    - Easy to modify chart appearance globally
    """
    
    @staticmethod
    def create_price_chart(data, predictions=None, title="Stock Price Chart", algorithm_name=None):
        """
        Create an enhanced interactive price chart with better UX.
        
        Args:
            data (pd.DataFrame): Historical stock data
            predictions (list): Optional prediction data
            title (str): Chart title
            algorithm_name (str): Algorithm used for predictions
            
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        try:
            # Create subplot with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.8, 0.2],
                subplot_titles=(None, "Volume")
            )
            
            # Add historical price line with hover information
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#2E86AB', width=3),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Price: ₹%{y:,.2f}<br>' +
                             '<extra></extra>',
            ), row=1, col=1)
            
            # Add high-low range as a filled area for better context
            if 'High' in data.columns and 'Low' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['High'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name='High'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Low'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='Price Range',
                    fillcolor='rgba(46, 134, 171, 0.1)',
                    hovertemplate='<b>Price Range</b><br>' +
                                 'High: ₹%{customdata[0]:,.2f}<br>' +
                                 'Low: ₹%{y:,.2f}<br>' +
                                 '<extra></extra>',
                    customdata=data[['High']].values
                ), row=1, col=1)
            
            # Add moving averages with better styling
            if 'SMA' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['SMA'],
                    mode='lines',
                    name='Moving Average (20)',
                    line=dict(color='#A23B72', width=2, dash='dash'),
                    hovertemplate='<b>SMA 20</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: ₹%{y:,.2f}<br>' +
                                 '<extra></extra>',
                ), row=1, col=1)
            
            # Add 50-day MA if we have enough data
            if len(data) >= 50:
                sma_50 = data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=sma_50,
                    mode='lines',
                    name='Moving Average (50)',
                    line=dict(color='#F39C12', width=2, dash='dot'),
                    hovertemplate='<b>SMA 50</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: ₹%{y:,.2f}<br>' +
                                 '<extra></extra>',
                ), row=1, col=1)
            
            # Add enhanced predictions with confidence intervals
            if predictions and len(predictions) > 0:
                try:
                    pred_dates = []
                    pred_prices = []
                    confidence_levels = []
                    
                    for pred in predictions:
                        if isinstance(pred, dict) and 'Date' in pred and 'Predicted_Price' in pred:
                            pred_dates.append(pred['Date'])
                            pred_prices.append(pred['Predicted_Price'])
                            confidence_levels.append(pred.get('confidence', 0.8))
                    
                    if pred_dates and pred_prices:
                        # Sort predictions by date
                        pred_data = list(zip(pred_dates, pred_prices, confidence_levels))
                        pred_data.sort(key=lambda x: x[0])
                        pred_dates, pred_prices, confidence_levels = zip(*pred_data)
                        
                        # Connect with historical data
                        if len(data) > 0:
                            last_historical_date = data['Date'].iloc[-1]
                            last_historical_price = data['Close'].iloc[-1]
                            
                            # Create smooth transition
                            transition_dates = [last_historical_date] + list(pred_dates)
                            transition_prices = [last_historical_price] + list(pred_prices)
                            
                            # Main prediction line
                            pred_name = f'Predictions ({algorithm_name})' if algorithm_name else 'Predictions'
                            fig.add_trace(go.Scatter(
                                x=transition_dates,
                                y=transition_prices,
                                mode='lines+markers',
                                name=pred_name,
                                line=dict(color='#E74C3C', width=4),
                                marker=dict(size=10, color='#E74C3C', symbol='diamond'),
                                hovertemplate='<b>Prediction</b><br>' +
                                             'Date: %{x}<br>' +
                                             'Price: ₹%{y:,.2f}<br>' +
                                             'Confidence: %{customdata:.0%}<br>' +
                                             '<extra></extra>',
                                customdata=[0.8] + list(confidence_levels)
                            ), row=1, col=1)
                            
                            # Add confidence interval
                            if len(pred_prices) > 1:
                                # Calculate confidence bounds (±5% based on confidence)
                                upper_bounds = [p * (1 + (1-c) * 0.05) for p, c in zip(transition_prices, [0.8] + list(confidence_levels))]
                                lower_bounds = [p * (1 - (1-c) * 0.05) for p, c in zip(transition_prices, [0.8] + list(confidence_levels))]
                                
                                # Upper bound
                                fig.add_trace(go.Scatter(
                                    x=transition_dates,
                                    y=upper_bounds,
                                    fill=None,
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    showlegend=False,
                                    name='Upper Bound'
                                ), row=1, col=1)
                                
                                # Lower bound with fill
                                fig.add_trace(go.Scatter(
                                    x=transition_dates,
                                    y=lower_bounds,
                                    fill='tonexty',
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    name='Confidence Interval',
                                    fillcolor='rgba(231, 76, 60, 0.2)',
                                    hovertemplate='<b>Confidence Range</b><br>' +
                                                 'Upper: ₹%{customdata[0]:,.2f}<br>' +
                                                 'Lower: ₹%{y:,.2f}<br>' +
                                                 '<extra></extra>',
                                    customdata=list(zip(upper_bounds))
                                ), row=1, col=1)
                
                except Exception as pred_error:
                    print(f"Error adding predictions to chart: {str(pred_error)}")
            
            # Add volume chart
            if 'Volume' in data.columns:
                fig.add_trace(go.Bar(
                    x=data['Date'],
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(46, 134, 171, 0.6)',
                    hovertemplate='<b>Volume</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Volume: %{y:,.0f}<br>' +
                                 '<extra></extra>',
                ), row=2, col=1)
            
            # Enhanced layout with better styling
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': '#2C3E50', 'family': 'Arial Black'}
                },
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                xaxis2_title='Date',
                yaxis2_title='Volume',
                hovermode='x unified',
                showlegend=True,
                height=700,
                plot_bgcolor='rgba(248, 249, 250, 0.8)',
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.3)',
                    showgrid=True,
                    zeroline=False,
                    showspikes=True,
                    spikethickness=1,
                    spikecolor='#999',
                    spikemode='across'
                ),
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.3)',
                    showgrid=True,
                    zeroline=False,
                    showspikes=True,
                    spikethickness=1,
                    spikecolor='#999',
                    spikemode='across'
                ),
                xaxis2=dict(
                    gridcolor='rgba(128,128,128,0.3)',
                    showgrid=True
                ),
                yaxis2=dict(
                    gridcolor='rgba(128,128,128,0.3)',
                    showgrid=True
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(128, 128, 128, 0.3)',
                    borderwidth=1
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Add range selector buttons for better navigation
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="7D", step="day", stepmode="backward"),
                            dict(count=30, label="1M", step="day", stepmode="backward"),
                            dict(count=90, label="3M", step="day", stepmode="backward"),
                            dict(count=180, label="6M", step="day", stepmode="backward"),
                            dict(count=365, label="1Y", step="day", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        bgcolor='rgba(46, 134, 171, 0.1)',
                        bordercolor='rgba(46, 134, 171, 0.3)',
                        borderwidth=1,
                        font=dict(color='#2C3E50')
                    ),
                    rangeslider=dict(visible=False),
                    type="date"
                )
            )
            
            # Add crossfilter cursor for better interaction
            fig.update_layout(
                dragmode='zoom',
                selectdirection='d'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating price chart: {str(e)}")
            return None
    
    @staticmethod
    def create_prediction_comparison_chart(actual_data, predictions, title="Prediction vs Actual"):
        """
        Create a chart comparing predictions with actual values.
        
        Args:
            actual_data (pd.DataFrame): Actual stock data
            predictions (list): Prediction data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Comparison chart
        """
        try:
            fig = go.Figure()
            
            # Add actual prices
            fig.add_trace(go.Scatter(
                x=actual_data['Date'],
                y=actual_data['Close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Add predictions
            if predictions:
                pred_dates = [pred['Date'] for pred in predictions]
                pred_prices = [pred['Predicted_Price'] for pred in predictions]
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='#F18F01', width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                hovermode='x unified',
                showlegend=True,
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating comparison chart: {str(e)}")
            return None
    
    @staticmethod
    def create_metrics_chart(metrics, title="Prediction Accuracy Metrics"):
        """
        Create a chart showing prediction accuracy metrics.
        
        Args:
            metrics (dict): Accuracy metrics
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Metrics chart
        """
        try:
            if 'error' in metrics:
                st.error(f"Cannot create metrics chart: {metrics['error']}")
                return None
            
            # Extract numeric metrics
            metric_names = []
            metric_values = []
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_names.append(key)
                    metric_values.append(value)
            
            if not metric_names:
                st.warning("No valid metrics to display")
                return None
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title='Metrics',
                yaxis_title='Value',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating metrics chart: {str(e)}")
            return None

class DataUtils:
    """
    Utility class for data processing and formatting.
    """
    
    @staticmethod
    def format_currency(value, currency="INR"):
        """Format a number as currency."""
        if pd.isna(value) or value is None:
            return "N/A"
        
        if currency == "INR":
            return f"₹{value:,.2f}"
        else:
            return f"${value:,.2f}"
    
    @staticmethod
    def format_percentage(value):
        """Format a number as percentage."""
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value:.2f}%"
    
    @staticmethod
    def format_number(value):
        """Format a large number with K, M, B suffixes."""
        if pd.isna(value) or value is None:
            return "N/A"
        
        if value >= 1e9:
            return f"{value/1e9:.1f}B"
        elif value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.0f}"
    
    @staticmethod
    def calculate_price_change_percentage(current_price, previous_price):
        """Calculate percentage change in price."""
        if pd.isna(current_price) or pd.isna(previous_price) or previous_price == 0:
            return None
        
        return ((current_price - previous_price) / previous_price) * 100
    
    @staticmethod
    def get_trend_indicator(change):
        """Get trend indicator emoji based on price change."""
        if pd.isna(change) or change is None:
            return "➡️"
        elif change > 0:
            return "📈"
        elif change < 0:
            return "📉"
        else:
            return "➡️"
    
    @staticmethod
    def detect_currency_from_symbol(symbol):
        """Always return INR for Indian market focus."""
        return "INR"

class ValidationUtils:
    """
    Utility class for input validation.
    """
    
    @staticmethod
    def validate_stock_symbol(symbol):
        """
        Validate stock symbol format.
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not symbol:
            return False, "Stock symbol cannot be empty"
        
        symbol = symbol.strip().upper()
        
        if len(symbol) < 1 or len(symbol) > 15:
            return False, "Stock symbol must be 1-15 characters long"
        
        # Allow alphanumeric characters and dots for Indian stocks (.NS)
        if not all(c.isalnum() or c in '.' for c in symbol):
            return False, "Stock symbol must contain only letters, numbers, and dots"
        
        return True, None
    
    @staticmethod
    def validate_prediction_parameters(window, predict_days):
        """
        Validate prediction parameters.
        
        Args:
            window (int): Moving average window
            predict_days (int): Days to predict
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if window < 5:
            return False, "Moving average window must be at least 5 days"
        
        if window > 200:
            return False, "Moving average window cannot exceed 200 days"
        
        if predict_days < 1:
            return False, "Prediction days must be at least 1"
        
        if predict_days > 365:
            return False, "Prediction days cannot exceed 365 (1 year)"
        
        return True, None

class PerformanceUtils:
    """
    Utility class for performance monitoring and optimization.
    """
    
    @staticmethod
    def get_system_health():
        """Get basic system health metrics."""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            return {
                'cpu_percent': 'N/A',
                'memory_percent': 'N/A',
                'disk_usage': 'N/A'
            }
    
    @staticmethod
    def cache_key_generator(func_name, *args, **kwargs):
        """Generate cache key for function calls."""
        import hashlib
        
        key_parts = [func_name] + [str(arg) for arg in args]
        key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
        key_string = "|".join(key_parts)
        
        return hashlib.md5(key_string.encode()).hexdigest()
