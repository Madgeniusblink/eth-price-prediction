#!/usr/bin/env python3
"""
Basic Prediction Example
Demonstrates simple usage of the prediction system
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import json

def main():
    """
    Run a basic prediction
    """
    print("=== Basic Ethereum Price Prediction Example ===\n")
    
    # Step 1: Fetch data
    print("Step 1: Fetching latest data...")
    import fetch_data
    print("✓ Data fetched\n")
    
    # Step 2: Generate predictions
    print("Step 2: Generating predictions...")
    import predict
    print("✓ Predictions generated\n")
    
    # Step 3: Read and display results
    print("Step 3: Reading prediction results...\n")
    
    try:
        with open('/home/ubuntu/predictions_summary.json', 'r') as f:
            results = json.load(f)
        
        print("=" * 50)
        print(f"Current Price: ${results['current_price']:.2f}")
        print(f"Market Trend: {results['trend_analysis']['trend']}")
        print("=" * 50)
        print("\nPrice Predictions:")
        print("-" * 50)
        
        for time_label, pred_data in results['predictions'].items():
            print(f"{time_label:>5}: ${pred_data['price']:>8.2f}  ({pred_data['change_pct']:>+6.2f}%)")
        
        print("-" * 50)
        print("\nModel Performance (R² Scores):")
        for model, score in results['model_scores'].items():
            print(f"  {model:>15}: {score:.4f}")
        
        print("\n✓ Prediction complete!")
        
    except FileNotFoundError:
        print("✗ Prediction file not found. Please run the prediction first.")
    except Exception as e:
        print(f"✗ Error reading results: {e}")

if __name__ == '__main__':
    main()
