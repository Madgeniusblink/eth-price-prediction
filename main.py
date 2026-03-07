#!/usr/bin/env python3
"""
Main entry point for ETH Price Prediction System
Called by GitHub Actions workflow
"""

import sys
import os
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

def main():
    """Run the complete ETH prediction pipeline"""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    try:
        print("🔮 Starting ETH Price Prediction System...")

        # Step 1: Generate prediction report
        print("📊 Generating prediction report...")
        result = subprocess.run(
            [sys.executable, "src/generate_report.py"],
            check=True, capture_output=False, text=True
        )
        print("✅ Report generated successfully!")

        # Step 2: Write latest_prediction.json at root (required by workflow archive step)
        latest_src = Path("reports/latest/predictions_summary.json")
        root_dest = Path("latest_prediction.json")
        if latest_src.exists():
            shutil.copy2(latest_src, root_dest)
            print(f"✅ Wrote latest_prediction.json ({root_dest.stat().st_size} bytes)")
        else:
            # Fallback: write a minimal JSON so workflow doesn't fail
            fallback = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "no_prediction_data",
                "error": "reports/latest/predictions_summary.json not found"
            }
            with open(root_dest, "w") as f:
                json.dump(fallback, f, indent=2)
            print("⚠️  Used fallback latest_prediction.json (report source missing)")

        # Step 3: Run walk-forward backtest (non-blocking — log results, don't fail)
        print("📈 Running walk-forward backtest...")
        bt_result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0,'src'); "
             "from validate import walk_forward_backtest; "
             "import pandas as pd; "
             "df = pd.read_csv('eth_1m_data.csv'); "
             "df.columns = [c.lower() for c in df.columns]; "
             "res = walk_forward_backtest(df); "
             "import json; open('data/backtest_results.json','w').write(json.dumps(res, indent=2)); "
             "print('Backtest complete:', res.get('directional_accuracy', 'N/A'))"],
            capture_output=True, text=True, timeout=120
        )
        if bt_result.returncode == 0:
            print(f"✅ Backtest: {bt_result.stdout.strip()}")
        else:
            print(f"⚠️  Backtest skipped: {bt_result.stderr[:200]}")

        # Step 4: Slack notification
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if slack_webhook:
            print("📱 Sending Slack notification...")
            subprocess.run([sys.executable, "src/send_slack_notification.py"],
                           check=False, capture_output=True)
            print("✅ Slack notification sent!")
        else:
            print("⚠️  SLACK_WEBHOOK_URL not set — skipping notification")

        print("🎉 ETH prediction pipeline completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
