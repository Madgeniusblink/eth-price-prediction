#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for Ethereum Price Prediction System
Runs the complete prediction workflow
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def main():
    """
    Main pipeline execution
    """
    print_header("ETHEREUM PRICE PREDICTION SYSTEM")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Fetch Data
    print_header("STEP 1: DATA COLLECTION")
    try:
        import fetch_data
        print("✓ Data collection completed successfully\n")
    except Exception as e:
        print(f"✗ Error in data collection: {e}")
        return 1
    
    # Step 2: Generate Predictions
    print_header("STEP 2: PREDICTION GENERATION")
    try:
        import predict
        print("✓ Predictions generated successfully\n")
    except Exception as e:
        print(f"✗ Error in prediction generation: {e}")
        return 1
    
    # Step 3: Create Visualizations
    print_header("STEP 3: VISUALIZATION")
    try:
        import visualize
        print("✓ Visualizations created successfully\n")
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        return 1
    
    # Step 4: Model Validation (optional, can be skipped for quick runs)
    print_header("STEP 4: MODEL VALIDATION (Optional)")
    response = input("Run model validation? This may take a few minutes. (y/n): ")
    
    if response.lower() == 'y':
        try:
            import validate
            print("✓ Validation completed successfully\n")
        except Exception as e:
            print(f"✗ Error in validation: {e}")
            print("Continuing without validation...\n")
    else:
        print("Skipping validation...\n")
    
    # Summary
    print_header("PIPELINE COMPLETED")
    print("Output files:")
    print(f"  - Data: {os.path.join(BASE_DIR, 'eth_1m_data.csv')}")
    print(f"  - Predictions: {os.path.join(BASE_DIR, 'predictions_summary.json')}")
    print("  - Visualizations:")
    print(f"    • {os.path.join(BASE_DIR, 'eth_prediction_overview.png')}")
    print(f"    • {os.path.join(BASE_DIR, 'eth_1hour_prediction.png')}")
    print(f"    • {os.path.join(BASE_DIR, 'eth_technical_indicators.png')}")
    print("\nCheck the files above for detailed results.")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
