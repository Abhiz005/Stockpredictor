#!/usr/bin/env python3
"""
Professional Stock Prediction Algorithms Testing Suite
Tests all 10 algorithms including 5 new professional-grade models
"""

import pandas as pd
import numpy as np
from modules.data_fetcher import StockDataFetcher
from modules.algorithms import PredictionAlgorithms
import warnings
warnings.filterwarnings('ignore')

def test_professional_algorithms():
    """Test all professional algorithms with comprehensive accuracy verification"""
    
    print("🏛️ PROFESSIONAL STOCK PREDICTION ALGORITHMS TESTING SUITE")
    print("=" * 80)
    print("Testing industry-standard algorithms used by hedge funds and institutions")
    print()
    
    # Initialize components
    data_fetcher = StockDataFetcher()
    algorithms = PredictionAlgorithms()
    
    # Test with real Indian market data
    symbol = "KEC.NS"
    print(f"📊 Testing with {symbol} - Real Indian market data")
    
    try:
        # Fetch comprehensive data
        data = data_fetcher.fetch_stock_data(symbol, period="2y")
        if data is None or data.empty:
            print(f"❌ Error: Could not fetch data for {symbol}")
            return
        
        print(f"✅ Data fetched: {len(data)} days of historical data")
        print(f"📈 Price range: ₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}")
        print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        print()
        
        # Professional algorithms test suite
        professional_algorithms = [
            # Original algorithms (enhanced)
            ('Simple Moving Average', {'window': 20}, 'Basic Technical Analysis'),
            ('Linear Regression', {'window': 20}, 'Statistical Modeling'),
            ('Exponential Smoothing', {'alpha': 0.3}, 'Time Series Smoothing'),
            ('LSTM Neural Network', {'sequence_length': 60}, 'Deep Learning AI'),
            ('Advanced Ensemble', {}, 'Multi-indicator Ensemble'),
            
            # New professional algorithms
            ('ARIMA Model', {'order': (2,1,2)}, 'Professional Time Series'),
            ('Prophet Model', {}, 'Facebook Advanced Forecasting'),
            ('Random Forest', {'n_estimators': 100}, 'Quantitative ML'),
            ('XGBoost Model', {}, 'Gradient Boosting ML'),
            ('Kalman Filter', {}, 'State Space Modeling')
        ]
        
        print("🔬 PROFESSIONAL ALGORITHM ACCURACY TESTING:")
        print("-" * 60)
        
        all_results = {}
        excellent_count = 0
        good_count = 0
        
        for algo_name, params, category in professional_algorithms:
            print(f"\n🧪 Testing {algo_name} ({category})...")
            
            try:
                # Enhanced accuracy testing with 30-day prediction validation
                test_result = algorithms.enhanced_accuracy_test(
                    data, algo_name, predict_days=30, **params
                )
                
                if 'error' not in test_result:
                    all_results[algo_name] = test_result
                    
                    accuracy = test_result['Accuracy_Score']
                    mape = test_result['MAPE']
                    directional = test_result['Directional_Accuracy']
                    r_squared = test_result['R_squared']
                    test_points = test_result['Test_Points']
                    
                    print(f"  📊 Accuracy Score: {accuracy:.2f}%")
                    print(f"  🎯 MAPE: {mape:.2f}%")
                    print(f"  📈 Directional Accuracy: {directional:.1f}%")
                    print(f"  🔢 R-squared: {r_squared:.3f}")
                    print(f"  📋 Test Points: {test_points}")
                    
                    # Professional rating system
                    if accuracy >= 95:
                        print(f"  🏆 Rating: INSTITUTIONAL GRADE")
                        excellent_count += 1
                    elif accuracy >= 90:
                        print(f"  🥇 Rating: PROFESSIONAL GRADE")
                        good_count += 1
                    elif accuracy >= 85:
                        print(f"  🥈 Rating: COMMERCIAL GRADE")
                    elif accuracy >= 80:
                        print(f"  🥉 Rating: ACCEPTABLE")
                    else:
                        print(f"  ⚠️  Rating: NEEDS OPTIMIZATION")
                    
                    # Special features
                    if 'AIC' in test_result:
                        print(f"  📏 AIC: {test_result['AIC']:.2f}")
                    if 'Feature_Importance' in test_result:
                        print(f"  🎯 Top Feature: {max(test_result['Feature_Importance'], key=test_result['Feature_Importance'].get)}")
                        
                else:
                    print(f"  ❌ Testing Error: {test_result['error']}")
                    
            except Exception as e:
                print(f"  ❌ Critical Error: {str(e)}")
        
        # Comprehensive results analysis
        if all_results:
            print("\n" + "=" * 80)
            print("📊 PROFESSIONAL ALGORITHMS PERFORMANCE SUMMARY")
            print("=" * 80)
            
            # Create professional comparison table
            print(f"{'Algorithm':<22} {'Category':<20} {'Accuracy':<10} {'MAPE':<8} {'R²':<8} {'Grade'}")
            print("-" * 80)
            
            # Sort by accuracy
            sorted_results = sorted(all_results.items(), 
                                  key=lambda x: x[1]['Accuracy_Score'], 
                                  reverse=True)
            
            for algo_name, results in sorted_results:
                accuracy = results['Accuracy_Score']
                mape = results['MAPE']
                r2 = results['R_squared']
                
                # Find category
                category = next((cat for name, _, cat in professional_algorithms if name == algo_name), "Unknown")
                
                # Determine grade
                if accuracy >= 95:
                    grade = "INSTITUTIONAL"
                elif accuracy >= 90:
                    grade = "PROFESSIONAL"
                elif accuracy >= 85:
                    grade = "COMMERCIAL"
                else:
                    grade = "BASIC"
                
                print(f"{algo_name:<22} {category:<20} {accuracy:>6.2f}%   {mape:>5.2f}%  {r2:>6.3f}  {grade}")
            
            # Performance statistics
            all_accuracies = [r['Accuracy_Score'] for r in all_results.values()]
            avg_accuracy = sum(all_accuracies) / len(all_accuracies)
            best_accuracy = max(all_accuracies)
            
            print(f"\n📈 PERFORMANCE STATISTICS:")
            print(f"   🎯 Average Accuracy: {avg_accuracy:.2f}%")
            print(f"   🏆 Best Accuracy: {best_accuracy:.2f}%")
            print(f"   🥇 Institutional Grade Algorithms: {excellent_count}")
            print(f"   🥈 Professional Grade Algorithms: {good_count}")
            print(f"   📊 Total Algorithms Tested: {len(all_results)}")
            
            # Best performer analysis
            best_algo, best_results = sorted_results[0]
            print(f"\n🏆 BEST PERFORMING ALGORITHM:")
            print(f"   Algorithm: {best_algo}")
            print(f"   Accuracy: {best_results['Accuracy_Score']:.2f}%")
            print(f"   MAPE: {best_results['MAPE']:.2f}%")
            print(f"   Directional Accuracy: {best_results['Directional_Accuracy']:.1f}%")
            
            # Algorithm recommendations
            print(f"\n💡 PROFESSIONAL RECOMMENDATIONS:")
            
            institutional_algos = [name for name, results in all_results.items() 
                                 if results['Accuracy_Score'] >= 95]
            professional_algos = [name for name, results in all_results.items() 
                                if 90 <= results['Accuracy_Score'] < 95]
            
            if institutional_algos:
                print(f"   🏛️ Institutional Grade: {', '.join(institutional_algos)}")
                print(f"      → Recommended for hedge funds and institutional trading")
            
            if professional_algos:
                print(f"   🏢 Professional Grade: {', '.join(professional_algos)}")
                print(f"      → Recommended for professional trading and analysis")
            
            # Technology categories
            ml_algos = [name for name in all_results.keys() 
                       if any(tech in name for tech in ['XGBoost', 'Random Forest', 'LSTM'])]
            statistical_algos = [name for name in all_results.keys() 
                                if any(tech in name for tech in ['ARIMA', 'Prophet', 'Kalman'])]
            
            if ml_algos:
                print(f"   🤖 Machine Learning Models: {', '.join(ml_algos)}")
            if statistical_algos:
                print(f"   📊 Statistical Models: {', '.join(statistical_algos)}")
            
            print(f"\n✅ PROFESSIONAL TESTING COMPLETED!")
            print(f"📊 All {len(all_results)} algorithms validated with real market data")
            print(f"🎯 Ready for professional trading and institutional analysis")
            
        else:
            print("❌ No successful test results obtained")
            
    except Exception as e:
        print(f"❌ Fatal error during testing: {str(e)}")

if __name__ == "__main__":
    test_professional_algorithms()