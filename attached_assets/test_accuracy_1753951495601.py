#!/usr/bin/env python3
"""
Comprehensive Algorithm Accuracy Testing Script
Tests all 5 enhanced algorithms and provides detailed accuracy metrics
"""

import pandas as pd
import numpy as np
from modules.data_fetcher import StockDataFetcher
from modules.algorithms import PredictionAlgorithms
import sys

def test_algorithm_accuracy():
    """Test all algorithms with real data and provide comprehensive results"""
    
    # Initialize components
    print("🚀 Starting Comprehensive Algorithm Accuracy Testing...")
    print("=" * 60)
    
    data_fetcher = StockDataFetcher()
    algorithms = PredictionAlgorithms()
    
    # Test with KEC.NS (user's preferred stock)
    symbol = "KEC.NS"
    print(f"📊 Testing with {symbol} - Real Indian market data")
    
    try:
        # Fetch real data
        data = data_fetcher.fetch_stock_data(symbol, period="2y")
        if data is None or data.empty:
            print(f"❌ Error: Could not fetch data for {symbol}")
            return
        
        print(f"✅ Data fetched: {len(data)} days of historical data")
        print(f"📈 Price range: ₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}")
        print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        print()
        
        # Test each algorithm
        all_results = {}
        
        algorithms_to_test = [
            ('Simple Moving Average', {'window': 20}),
            ('Linear Regression', {'window': 20}),
            ('Exponential Smoothing', {'alpha': 0.3}),
            ('LSTM Neural Network', {'sequence_length': 60}),
            ('Advanced Ensemble', {})
        ]
        
        print("🔬 Testing Individual Algorithms:")
        print("-" * 40)
        
        for algo_name, params in algorithms_to_test:
            print(f"\n🧪 Testing {algo_name}...")
            
            try:
                # Enhanced accuracy testing
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
                    
                    print(f"  ✅ Accuracy Score: {accuracy:.2f}%")
                    print(f"  📊 MAPE: {mape:.2f}%")
                    print(f"  🎯 Directional Accuracy: {directional:.1f}%")
                    print(f"  📈 R-squared: {r_squared:.3f}")
                    print(f"  🔢 Test Points: {test_points}")
                    
                    # Performance rating
                    if accuracy >= 95:
                        print(f"  🏆 Rating: EXCELLENT")
                    elif accuracy >= 90:
                        print(f"  🥇 Rating: VERY GOOD")
                    elif accuracy >= 85:
                        print(f"  🥈 Rating: GOOD")
                    elif accuracy >= 80:
                        print(f"  🥉 Rating: ACCEPTABLE")
                    else:
                        print(f"  ⚠️  Rating: NEEDS IMPROVEMENT")
                        
                else:
                    print(f"  ❌ Error: {test_result['error']}")
                    
            except Exception as e:
                print(f"  ❌ Testing failed: {str(e)}")
        
        # Summary and comparison
        if all_results:
            print("\n" + "=" * 60)
            print("📊 COMPREHENSIVE RESULTS SUMMARY")
            print("=" * 60)
            
            # Create comparison table
            print(f"{'Algorithm':<25} {'Accuracy':<12} {'MAPE':<8} {'Dir.Acc':<8} {'R²':<8}")
            print("-" * 65)
            
            sorted_results = sorted(all_results.items(), 
                                  key=lambda x: x[1]['Accuracy_Score'], 
                                  reverse=True)
            
            for algo_name, results in sorted_results:
                accuracy = results['Accuracy_Score']
                mape = results['MAPE']
                dir_acc = results['Directional_Accuracy']
                r2 = results['R_squared']
                
                print(f"{algo_name:<25} {accuracy:>8.2f}%   {mape:>5.2f}%  {dir_acc:>5.1f}%  {r2:>6.3f}")
            
            # Best performer
            best_algo, best_results = sorted_results[0]
            print(f"\n🏆 BEST PERFORMER: {best_algo}")
            print(f"   Accuracy: {best_results['Accuracy_Score']:.2f}%")
            print(f"   MAPE: {best_results['MAPE']:.2f}%")
            print(f"   Directional Accuracy: {best_results['Directional_Accuracy']:.1f}%")
            
            # Overall statistics
            all_accuracies = [r['Accuracy_Score'] for r in all_results.values()]
            avg_accuracy = sum(all_accuracies) / len(all_accuracies)
            
            print(f"\n📈 OVERALL STATISTICS:")
            print(f"   Average Accuracy: {avg_accuracy:.2f}%")
            print(f"   Best Accuracy: {max(all_accuracies):.2f}%")
            print(f"   Algorithms > 90%: {sum(1 for acc in all_accuracies if acc >= 90)}/{len(all_accuracies)}")
            print(f"   Algorithms > 95%: {sum(1 for acc in all_accuracies if acc >= 95)}/{len(all_accuracies)}")
            
            # Enhancement recommendations
            print(f"\n🔧 ENHANCEMENT RECOMMENDATIONS:")
            for algo_name, results in all_results.items():
                accuracy = results['Accuracy_Score']
                if accuracy < 90:
                    improvement = 95 - accuracy
                    print(f"   {algo_name}: Improve by {improvement:.1f}% to reach excellent rating")
            
            print(f"\n✅ Testing completed successfully!")
            print(f"📊 All {len(all_results)} algorithms tested with real market data")
            
        else:
            print("❌ No successful test results obtained")
            
    except Exception as e:
        print(f"❌ Fatal error during testing: {str(e)}")

if __name__ == "__main__":
    test_algorithm_accuracy()