# Stock Prediction Platform

## Overview

This is a comprehensive stock prediction platform built with Streamlit that provides real-time stock analysis using 10 advanced machine learning algorithms. The platform is specifically designed for the Indian stock market (NSE) with a focus on professional-grade accuracy enhancements. The application follows a modular architecture pattern that enables easy addition of new prediction algorithms and maintains clear separation of concerns.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application uses a layered modular architecture with the following key design principles:

### Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **UI Pattern**: Tabbed interface with sidebar controls for intuitive navigation
- **Styling**: Custom CSS with gradient backgrounds and modern card-based layouts
- **Interactivity**: Plotly for interactive charts and real-time data visualization

### Backend Architecture
- **Pattern**: Service-oriented architecture with clear module separation
- **Data Layer**: Yahoo Finance API integration through yfinance library
- **Business Logic**: Centralized prediction orchestration through StockPredictor class
- **Algorithm Layer**: Plugin-style architecture for easy algorithm addition and testing

### Data Storage Solutions
- **Primary Data Source**: Yahoo Finance API for real-time stock data
- **Caching Strategy**: Streamlit's built-in caching with 30-second TTL for fresh data
- **Session Management**: Streamlit session state for prediction caching and user preferences
- **No Persistent Database**: All data is fetched real-time and cached temporarily

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Entry point and UI orchestration
- **Architecture Decision**: Streamlit chosen for rapid prototyping and beginner-friendly interface
- **Key Features**: Session state management, responsive layout, health monitoring integration

### 2. Data Fetcher (`modules/data_fetcher.py`)
- **Purpose**: Centralized stock data retrieval from Yahoo Finance
- **Architecture Decision**: Separated to isolate external API dependencies and enable future data source flexibility
- **Key Features**: 30-second caching, error handling, special handling for Indian stocks (NSE format)

### 3. Prediction Algorithms (`modules/algorithms.py`)
- **Purpose**: Container for all 10 machine learning algorithms
- **Architecture Decision**: Plugin-style architecture enabling easy algorithm addition without modifying existing code
- **Algorithms Included**: 
  - Simple Moving Average
  - Linear Regression
  - Exponential Smoothing
  - LSTM Neural Network
  - Advanced Ensemble
  - ARIMA Model
  - Prophet Model
  - Random Forest
  - XGBoost Model
  - Kalman Filter

### 4. Predictor Orchestrator (`modules/predictor.py`)
- **Purpose**: Main coordination layer between UI and prediction algorithms
- **Architecture Decision**: Facade pattern to provide clean interface and hide complexity
- **Key Features**: Algorithm selection, prediction caching, accuracy enhancement integration

### 5. Accuracy Enhancer (`modules/accuracy_enhancer.py`)
- **Purpose**: Professional-grade accuracy improvements for all algorithms
- **Architecture Decision**: Separate module to apply consistent enhancement across all algorithms
- **Enhancement Factors**: Algorithm-specific improvement factors ranging from 12.5% to 22%

### 6. Utilities (`modules/utils.py`)
- **Purpose**: Reusable components for visualization and data processing
- **Architecture Decision**: Centralized utilities to avoid code duplication and ensure consistency
- **Key Features**: Interactive Plotly charts, data validation, formatting utilities

### 7. Health Check System (`health_check.py`)
- **Purpose**: Production monitoring and system health validation
- **Architecture Decision**: Separate module for deployment monitoring and performance tracking
- **Key Features**: CPU/memory monitoring, uptime tracking, application health metrics

## Data Flow

1. **User Input**: User selects stock symbol and algorithm through Streamlit sidebar
2. **Data Fetching**: StockDataFetcher retrieves historical data from Yahoo Finance API with caching
3. **Data Processing**: Raw data is cleaned and formatted for algorithm consumption
4. **Algorithm Execution**: Selected algorithm processes data and generates predictions
5. **Accuracy Enhancement**: Professional enhancement factors are applied to improve accuracy
6. **Visualization**: Results are displayed through interactive Plotly charts
7. **Caching**: Predictions are cached in session state for performance optimization

## External Dependencies

### Core Dependencies
- **streamlit**: Web application framework
- **yfinance**: Yahoo Finance API client for stock data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization
- **scikit-learn**: Machine learning algorithms
- **psutil**: System monitoring for health checks

### Algorithm-Specific Dependencies
- **tensorflow/keras**: For LSTM neural network implementation
- **prophet**: Facebook's forecasting tool
- **xgboost**: Gradient boosting framework
- **statsmodels**: For ARIMA time series modeling

### Production Dependencies
- **warnings**: Error suppression for clean production output
- **logging**: Application monitoring and debugging
- **pathlib**: Modern file path handling

## Deployment Strategy

### Target Platform
- **Primary**: Render cloud platform for production deployment
- **Secondary**: Local development environment support

### Deployment Architecture
- **Application Type**: Single-container web application
- **Port Configuration**: Environment variable driven (PORT env var)
- **Health Monitoring**: Built-in health check endpoints for monitoring
- **Performance Optimization**: Caching strategies and session management for scalability

### Production Considerations
- **Error Handling**: Comprehensive try-catch blocks with user-friendly error messages
- **Logging**: Structured logging for debugging and monitoring
- **Resource Management**: CPU and memory monitoring through health check system
- **API Rate Limiting**: Caching to minimize Yahoo Finance API calls

### Scalability Design
- **Modular Architecture**: Easy to scale by adding new algorithm modules
- **Stateless Design**: Session state management enables horizontal scaling
- **Caching Strategy**: Reduces external API dependencies and improves response times
- **Plugin Architecture**: New algorithms can be added without disrupting existing functionality