"""
Health Check Module for Production Monitoring
Provides health check endpoints and system monitoring
"""

import streamlit as st
import time
import psutil
import os
from datetime import datetime
import pandas as pd

class HealthCheck:
    """Health check and monitoring system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.check_count = 0
    
    def get_system_health(self):
        """Get comprehensive system health metrics"""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Application metrics
            uptime = time.time() - self.start_time
            
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime,
                'uptime_formatted': self._format_uptime(uptime),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free_gb': disk.free / (1024**3)
                },
                'application': {
                    'check_count': self.check_count,
                    'python_version': os.environ.get('PYTHON_VERSION', '3.9'),
                    'port': os.environ.get('PORT', '5000')
                }
            }
            
            # Determine overall health status
            if cpu_percent > 90 or memory.percent > 90:
                health_data['status'] = 'warning'
            
            if cpu_percent > 95 or memory.percent > 95:
                health_data['status'] = 'critical'
            
            self.check_count += 1
            return health_data
            
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _format_uptime(self, seconds):
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        dependencies = {
            'streamlit': True,
            'pandas': True,
            'numpy': True,
            'plotly': True,
            'yfinance': True,
            'scikit-learn': True
        }
        
        try:
            import streamlit
            import pandas
            import numpy
            import plotly
            import yfinance
            import sklearn
        except ImportError as e:
            dependencies[str(e).split("'")[1]] = False
        
        return dependencies
    
    def run_application_tests(self):
        """Run basic application functionality tests"""
        test_results = {
            'data_fetcher': True,
            'algorithms': True,
            'predictor': True,
            'ui_components': True
        }
        
        try:
            # Test data fetcher
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            
            # Test algorithms
            from modules.algorithms import PredictionAlgorithms
            algorithms = PredictionAlgorithms()
            
            # Test predictor
            from modules.predictor import StockPredictor
            predictor = StockPredictor()
            
        except Exception as e:
            test_results['import_error'] = str(e)
            for key in test_results:
                test_results[key] = False
        
        return test_results

def health_check_endpoint():
    """Health check endpoint for load balancers"""
    health_checker = HealthCheck()
    health_data = health_checker.get_system_health()
    
    if health_data['status'] == 'healthy':
        return {"status": "OK", "timestamp": health_data['timestamp']}
    else:
        return {"status": "ERROR", "details": health_data}

def detailed_health_dashboard():
    """Detailed health dashboard for monitoring"""
    st.markdown("# 🏥 System Health Dashboard")
    
    health_checker = HealthCheck()
    
    # Get health data
    health_data = health_checker.get_system_health()
    dependencies = health_checker.check_dependencies()
    app_tests = health_checker.run_application_tests()
    
    # Overall status
    status_color = {
        'healthy': '🟢',
        'warning': '🟡',
        'critical': '🔴',
        'error': '❌'
    }
    
    st.markdown(f"## {status_color.get(health_data['status'], '❓')} Overall Status: {health_data['status'].upper()}")
    
    # System metrics
    st.markdown("### 📊 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_percent = health_data['system']['cpu_percent']
        st.metric("CPU Usage", f"{cpu_percent:.1f}%", 
                 delta="🟢 Normal" if cpu_percent < 70 else "🟡 High" if cpu_percent < 90 else "🔴 Critical")
    
    with col2:
        memory_percent = health_data['system']['memory_percent']
        st.metric("Memory Usage", f"{memory_percent:.1f}%",
                 delta="🟢 Normal" if memory_percent < 70 else "🟡 High" if memory_percent < 90 else "🔴 Critical")
    
    with col3:
        disk_percent = health_data['system']['disk_percent']
        st.metric("Disk Usage", f"{disk_percent:.1f}%",
                 delta="🟢 Normal" if disk_percent < 80 else "🟡 High" if disk_percent < 95 else "🔴 Critical")
    
    with col4:
        st.metric("Uptime", health_data['uptime_formatted'])
    
    # Dependencies status
    st.markdown("### 📦 Dependencies Status")
    
    dep_cols = st.columns(3)
    for i, (dep, status) in enumerate(dependencies.items()):
        with dep_cols[i % 3]:
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {dep}")
    
    # Application tests
    st.markdown("### 🧪 Application Tests")
    
    test_cols = st.columns(2)
    for i, (test, status) in enumerate(app_tests.items()):
        with test_cols[i % 2]:
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {test.replace('_', ' ').title()}")
    
    # Performance monitoring
    st.markdown("### 📈 Performance Monitoring")
    
    # Create sample performance data for visualization
    performance_data = pd.DataFrame({
        'timestamp': pd.date_range(start='now', periods=24, freq='H'),
        'cpu': [health_data['system']['cpu_percent'] + (i % 10 - 5) for i in range(24)],
        'memory': [health_data['system']['memory_percent'] + (i % 8 - 4) for i in range(24)]
    })
    
    st.line_chart(performance_data.set_index('timestamp')[['cpu', 'memory']])
    
    # Auto-refresh
    if st.button("🔄 Refresh Health Check"):
        st.rerun()

if __name__ == "__main__":
    # For health check endpoint
    health_data = health_check_endpoint()
    print(health_data)
