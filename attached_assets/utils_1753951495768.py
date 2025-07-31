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
    def create_price_chart(data, predictions=None, title="Stock Price Chart"):
        """
        Create an interactive price chart with optional predictions.
        
        Args:
            data (pd.DataFrame): Historical stock data
            predictions (list): Optional prediction data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        try:
            fig = go.Figure()
            
            # Add historical price line
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add moving average if available
            if 'SMA' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['SMA'],
                    mode='lines',
                    name='Moving Average',
                    line=dict(color='orange', width=2, dash='dash')
                ))
            
            # Add predictions if provided
            if predictions and len(predictions) > 0:
                try:
                    pred_dates = []
                    pred_prices = []
                    
                    for pred in predictions:
                        if isinstance(pred, dict) and 'Date' in pred and 'Predicted_Price' in pred:
                            pred_dates.append(pred['Date'])
                            pred_prices.append(pred['Predicted_Price'])
                    
                    if pred_dates and pred_prices:
                        # Sort predictions by date to ensure proper line connection
                        pred_data = list(zip(pred_dates, pred_prices))
                        pred_data.sort(key=lambda x: x[0])
                        pred_dates, pred_prices = zip(*pred_data)
                        
                        # Connect the last historical price to first prediction for smooth transition
                        if len(data) > 0:
                            last_historical_date = data['Date'].iloc[-1]
                            last_historical_price = data['Close'].iloc[-1]
                            
                            # Create smooth transition
                            transition_dates = [last_historical_date] + list(pred_dates)
                            transition_prices = [last_historical_price] + list(pred_prices)
                            
                            fig.add_trace(go.Scatter(
                                x=transition_dates,
                                y=transition_prices,
                                mode='lines+markers',
                                name='Predictions',
                                line=dict(color='red', width=3),
                                marker=dict(size=8, color='red')
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=pred_dates,
                                y=pred_prices,
                                mode='lines+markers',
                                name='Predictions',
                                line=dict(color='red', width=3),
                                marker=dict(size=8, color='red')
                            ))
                except Exception as pred_error:
                    print(f"Error adding predictions to chart: {str(pred_error)}")
                    # Continue without predictions
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                hovermode='x unified',
                showlegend=True,
                height=600
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
                line=dict(color='blue', width=2)
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
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                showlegend=True,
                height=500
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
                height=400
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
    def format_currency(value, currency="USD"):
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
