#!/usr/bin/env python3
"""
Main entry point for ETH Price Prediction System
Called by GitHub Actions workflow hourly.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def main():
    """Run the complete ETH prediction pipeline"""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    try:
        print("🔮 Starting ETH Price Prediction System...")

        # Step 1: Generate prediction report (fetches data, runs ML, saves JSON files)
        print("\n📊 Step 1/4: Generating prediction report...")
        result = subprocess.run(
            [sys.executable, "src/generate_report.py"],
            check=True, capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-2000:])
        print("✅ Report generated successfully!")

        # Step 2: Run derivatives data fetch (standalone — updates derivatives context)
        print("\n📈 Step 2/4: Fetching derivatives data (funding rate, OI, F&G)...")
        try:
            result2 = subprocess.run(
                [sys.executable, "-c",
                 "import sys; sys.path.insert(0,'src'); from derivatives_data import fetch_derivatives_data; "
                 "import json; d=fetch_derivatives_data(); print(json.dumps(d,indent=2))"],
                capture_output=True, text=True, timeout=30
            )
            if result2.returncode == 0:
                print(result2.stdout[:500])
                print("✅ Derivatives data fetched!")
            else:
                print(f"⚠️  Derivatives fetch warning: {result2.stderr[:300]}")
        except Exception as e:
            print(f"⚠️  Derivatives step skipped: {e}")

        # Step 3: Run accuracy tracker
        print("\n📐 Step 3/4: Running signal accuracy tracker...")
        try:
            result3 = subprocess.run(
                [sys.executable, "src/signal_accuracy_tracker.py"],
                capture_output=True, text=True, timeout=60
            )
            if result3.returncode == 0:
                print(result3.stdout[:500])
                print("✅ Accuracy tracker complete!")
            else:
                print(f"⚠️  Accuracy tracker warning: {result3.stderr[:300]}")
        except Exception as e:
            print(f"⚠️  Accuracy tracker skipped: {e}")

        # Step 4: Send Slack notification
        print("\n📱 Step 4/4: Sending Slack notification...")
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if not slack_webhook:
            print("⚠️  SLACK_WEBHOOK_URL not set — skipping notification")
        else:
            # Locate the output files (generate_report saves to BASE_DIR root)
            predictions_file = str(script_dir / "predictions_summary.json")
            signals_file     = str(script_dir / "trading_signals.json")
            report_url       = "https://github.com/Madgeniusblink/eth-price-prediction"

            # Fallback: check reports/latest if not in root
            if not os.path.exists(predictions_file):
                predictions_file = str(script_dir / "reports/latest/predictions_summary.json")
            if not os.path.exists(signals_file):
                signals_file = str(script_dir / "reports/latest/trading_signals.json")

            pred_ok    = os.path.exists(predictions_file)
            signals_ok = os.path.exists(signals_file)
            print(f"  predictions_summary.json : {'✓' if pred_ok else '✗'} ({predictions_file})")
            print(f"  trading_signals.json     : {'✓' if signals_ok else '✗'} ({signals_file})")

            if pred_ok and signals_ok:
                result4 = subprocess.run(
                    [sys.executable, "src/send_slack_notification.py",
                     predictions_file, signals_file, report_url],
                    check=True, capture_output=True, text=True
                )
                print(result4.stdout)
                print("✅ Slack notification sent!")
            else:
                # Try to send a minimal alert if files missing
                print("⚠️  Output files not found — sending minimal Slack alert")
                _send_minimal_alert(slack_webhook, "⚠️ ETH prediction ran but output files missing. Check logs.")

        # Create latest_prediction.json (alias for predictions_summary.json)
        # Required by the GitHub Actions archive step
        pred_src = script_dir / "predictions_summary.json"
        if pred_src.exists():
            import shutil
            shutil.copy(str(pred_src), str(script_dir / "latest_prediction.json"))

        print("\n🎉 ETH prediction pipeline completed successfully!")

    except subprocess.CalledProcessError as e:
        msg = f"❌ Pipeline step failed: {e}\nSTDOUT: {e.stdout[-1000:]}\nSTDERR: {e.stderr[-1000:]}"
        print(msg)
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if slack_webhook:
            _send_minimal_alert(slack_webhook, f"🚨 ETH Prediction FAILED\n```{e.stderr[-500:]}```")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if slack_webhook:
            _send_minimal_alert(slack_webhook, f"🚨 ETH Prediction FAILED (unexpected)\n{str(e)[:300]}")
        sys.exit(1)


def _send_minimal_alert(webhook_url: str, text: str):
    """Send a plain-text Slack alert without needing data files."""
    import urllib.request
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(webhook_url, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Could not send minimal alert: {e}")


if __name__ == "__main__":
    main()
# This is appended - do not use
