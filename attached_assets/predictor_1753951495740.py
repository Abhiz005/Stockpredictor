"""
Predictor Module
This module orchestrates the prediction process by combining data fetching and algorithms.
It's the main interface between the UI and the prediction logic.
"""

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from .data_fetcher import StockDataFetcher
from .algorithms import PredictionAlgorithms

class StockPredictor:
    """
    Main predictor class that orchestrates the prediction process.
    
    Why this class?
    - Separates prediction logic from UI
    - Makes it easy to change algorithms or data sources
    - Provides a clean interface for the main app
    """
    
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.algorithms = PredictionAlgorithms()
        self.enhancement_factors = {
            'Simple Moving Average': 12.5,
            'Linear Regression': 15.0,
            'Exponential Smoothing': 13.5,
            'LSTM Neural Network': 18.0,
            'Advanced Ensemble': 22.0,
            'ARIMA Model': 16.5,
            'Prophet Model': 19.0,
            'Random Forest': 17.0,
            'XGBoost Model': 20.0,
            'Kalman Filter': 15.5
        }
        self.prediction_cache = {}
    
    def predict_stock_price(self, symbol, algorithm_name, period="1y", **kwargs):
        """
        Main prediction method.
        
        Args:
            symbol (str): Stock symbol
            algorithm_name (str): Name of algorithm to use
            period (str): Historical data period
            **kwargs: Additional parameters for the algorithm
            
        Returns:
            dict: Prediction results
        """
        try:
            # Validate symbol first
            if not self.data_fetcher.validate_symbol(symbol):
                return {'error': f'Invalid stock symbol: {symbol}'}
            
            # Fetch historical data
            historical_data = self.data_fetcher.fetch_stock_data(symbol, period)
            
            if historical_data is None or historical_data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Get the algorithm function
            available_algorithms = self.algorithms.get_available_algorithms()
            
            if algorithm_name not in available_algorithms:
                return {'error': f'Algorithm {algorithm_name} not available'}
            
            algorithm_func = self.algorithms.algorithms[algorithm_name]
            
            # Run the prediction
            prediction_result = algorithm_func(historical_data, **kwargs)
            
            if prediction_result is None:
                return {'error': f'Prediction failed for {symbol}'}
            
            # Add metadata
            prediction_result['symbol'] = symbol
            prediction_result['algorithm'] = algorithm_name
            prediction_result['prediction_date'] = datetime.now()
            prediction_result['data_period'] = period
            
            return prediction_result
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def get_multiple_predictions(self, symbol, algorithms_list, period="1y", predict_days=5, algorithm_params=None, **kwargs):
        """
        Get predictions from multiple algorithms.
        
        Args:
            symbol (str): Stock symbol
            algorithms_list (list): List of algorithm names
            period (str): Historical data period
            **kwargs: Additional parameters
            
        Returns:
            dict: Combined prediction results
        """
        try:
            predictions = []
            individual_results = {}
            
            # Prepare algorithm parameters
            if algorithm_params is None:
                algorithm_params = {}
            
            for algorithm in algorithms_list:
                # Prepare parameters for each specific algorithm
                params = {'predict_days': predict_days}
                
                if algorithm == 'Simple Moving Average':
                    params['window'] = algorithm_params.get('sma_window', 20)
                elif algorithm == 'Linear Regression':
                    params['window'] = algorithm_params.get('lr_window', 20)
                elif algorithm == 'Exponential Smoothing':
                    params['alpha'] = algorithm_params.get('alpha', 0.3)
                elif algorithm == 'LSTM Neural Network':
                    params['sequence_length'] = algorithm_params.get('sequence_length', 60)
                
                result = self.predict_stock_price(symbol, algorithm, period, **params)
                
                if 'error' not in result:
                    predictions.append(result)
                    individual_results[algorithm] = result
                else:
                    st.warning(f"Algorithm {algorithm} failed: {result['error']}")
            
            if not predictions:
                return {'error': 'All algorithms failed to generate predictions'}
            
            # Combine predictions if more than one algorithm succeeded
            if len(predictions) > 1:
                combined_result = self.algorithms.combine_algorithms(predictions)
                
                return {
                    'combined_predictions': combined_result,
                    'individual_predictions': individual_results,
                    'symbol': symbol,
                    'algorithms_used': algorithms_list,
                    'prediction_date': datetime.now()
                }
            else:
                return {
                    'combined_predictions': predictions[0],
                    'individual_predictions': individual_results,
                    'symbol': symbol,
                    'algorithms_used': algorithms_list,
                    'prediction_date': datetime.now()
                }
                
        except Exception as e:
            return {'error': f'Multiple prediction error: {str(e)}'}
    
    def get_prediction_confidence(self, prediction_result):
        """
        Calculate confidence level for predictions.
        
        Why confidence levels?
        - Helps users understand prediction reliability
        - Important for risk management
        - Shows uncertainty in predictions
        
        Args:
            prediction_result (dict): Prediction results
            
        Returns:
            dict: Confidence metrics
        """
        try:
            if 'metrics' not in prediction_result:
                return {'confidence': 'Unknown', 'level': 0}
            
            metrics = prediction_result['metrics']
            
            if 'error' in metrics:
                return {'confidence': 'Error', 'level': 0}
            
            # Calculate confidence based on accuracy metrics
            accuracy = metrics.get('Accuracy_Score', 0)
            r_squared = metrics.get('R_squared', 0)
            
            # Simple confidence calculation - can be enhanced later
            confidence_score = (accuracy + (r_squared * 100)) / 2
            
            if confidence_score >= 80:
                confidence_level = 'High'
            elif confidence_score >= 60:
                confidence_level = 'Medium'
            elif confidence_score >= 40:
                confidence_level = 'Low'
            else:
                confidence_level = 'Very Low'
            
            return {
                'confidence': confidence_level,
                'level': confidence_score,
                'accuracy': accuracy,
                'r_squared': r_squared
            }
            
        except Exception as e:
            return {'confidence': 'Error', 'level': 0, 'error': str(e)}
    
    def get_stock_summary(self, symbol):
        """
        Get a comprehensive summary of stock information and predictions.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock summary
        """
        try:
            # Get basic stock info
            stock_info = self.data_fetcher.get_stock_info(symbol)
            
            # Get recent data for current price
            recent_data = self.data_fetcher.fetch_stock_data(symbol, "5d")
            
            summary = {
                'symbol': symbol,
                'info': stock_info,
                'current_price': None,
                'price_change': None,
                'volume': None
            }
            
            if recent_data is not None and not recent_data.empty:
                latest = recent_data.iloc[-1]
                previous = recent_data.iloc[-2] if len(recent_data) > 1 else latest
                
                summary['current_price'] = latest['Close']
                summary['price_change'] = latest['Close'] - previous['Close']
                summary['volume'] = latest['Volume']
            
            return summary
            
        except Exception as e:
            return {'error': f'Error getting stock summary: {str(e)}'}
