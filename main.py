#!/usr/bin/env python3
"""
Main entry point for ETH Price Prediction System
This file is called by GitHub Actions workflow
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Run the complete ETH prediction pipeline"""
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        print("🔮 Starting ETH Price Prediction System...")
        
        # Run the main report generation
        print("📊 Generating prediction report...")
        result = subprocess.run([
            sys.executable, "src/generate_report.py"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Report generated successfully!")
        print(result.stdout)
        
        # Send Slack notification if webhook is configured
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if slack_webhook:
            print("📱 Sending Slack notification...")
            
            # Define the expected output files
            predictions_file = "predictions_summary.json"
            signals_file = "trading_signals.json"  
            report_url = "https://github.com/Madgeniusblink/eth-price-prediction"
            
            # Check if the files exist
            if os.path.exists(predictions_file) and os.path.exists(signals_file):
                result = subprocess.run([
                    sys.executable, "src/send_slack_notification.py",
                    predictions_file, signals_file, report_url
                ], check=True, capture_output=True, text=True)
                print("✅ Slack notification sent!")
                print(result.stdout)
            else:
                print("⚠️ Required files not found for Slack notification:")
                print(f"  - {predictions_file}: {'✓' if os.path.exists(predictions_file) else '✗'}")
                print(f"  - {signals_file}: {'✓' if os.path.exists(signals_file) else '✗'}")
        else:
            print("⚠️  SLACK_WEBHOOK_URL not configured - skipping Slack notification")
        
        print("🎉 ETH prediction pipeline completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in prediction pipeline: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()