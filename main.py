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

        # Step 3.5: Run quant DeFi analysis (non-breaking)
        print("\n🔬 Step 3.5/4: Running quant DeFi analysis (volatility regime + LP optimizer)...")
        quant_data = None
        try:
            import sys as _sys
            _sys.path.insert(0, str(script_dir / 'src'))
            from fetch_data import fetch_quant_data
            from volatility_regime import get_volatility_regime
            from uniswap_optimizer import get_lp_recommendation
            from trading_signals import compute_signal_strength

            qd = fetch_quant_data()
            spot = qd.get('spot') or {}
            ohlc_df = qd.get('ohlc_df')
            defillama = qd.get('defillama')

            # Load existing signals for strength scoring
            signals_path = str(script_dir / "reports/latest/trading_signals.json")
            if not os.path.exists(signals_path):
                signals_path = str(script_dir / "trading_signals.json")
            with open(signals_path) as _f:
                _signals = json.load(_f)

            # Use 1m Kraken data for vol regime (better granularity than CoinGecko OHLC)
            import pandas as pd
            kraken_1m = str(script_dir / "eth_1m_data.csv")
            vol_df = None
            if os.path.exists(kraken_1m):
                vol_df = pd.read_csv(kraken_1m, parse_dates=['timestamp'])
            elif ohlc_df is not None:
                vol_df = ohlc_df

            regime_data = get_volatility_regime(vol_df) if vol_df is not None else {"regime": "UNKNOWN"}
            regime = regime_data.get("regime", "UNKNOWN")

            current_price = spot.get('price') or 0
            if not current_price:
                # fallback to predictions file
                pred_path = str(script_dir / "reports/latest/predictions_summary.json")
                if not os.path.exists(pred_path):
                    pred_path = str(script_dir / "predictions_summary.json")
                if os.path.exists(pred_path):
                    with open(pred_path) as _f:
                        _p = json.load(_f)
                    current_price = _p.get('current_price', 0)

            lp_data = get_lp_recommendation(current_price, regime, defillama) if current_price else {}
            strength = compute_signal_strength(_signals, regime)

            quant_data = {
                'spot': spot,
                'regime': regime_data,
                'lp': lp_data,
                'strength': strength,
            }

            # Save quant_data alongside predictions for send_slack_notification
            for _dir in [str(script_dir / "reports/latest"), str(script_dir)]:
                try:
                    _qf = os.path.join(_dir, 'quant_data.json')
                    with open(_qf, 'w') as _f:
                        json.dump(quant_data, _f, indent=2, default=str)
                except Exception:
                    pass

            print(f"✅ Quant analysis: regime={regime}, score={strength.get('score')}/100, LP={lp_data.get('recommendation')}")
        except Exception as e:
            import traceback
            print(f"⚠️  Quant analysis failed (non-critical, continuing): {e}")
            traceback.print_exc()

        # Step 4: Send Slack notification
        print("\n📱 Step 4/4: Sending Slack notification...")
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if not slack_webhook:
            print("⚠️  SLACK_WEBHOOK_URL not set — skipping notification")
        else:
            # Always use reports/latest/ — these are freshly generated each run.
            # Root-level trading_signals.json is stale (not committed by workflow).
            predictions_file = str(script_dir / "reports/latest/predictions_summary.json")
            signals_file     = str(script_dir / "reports/latest/trading_signals.json")
            report_url       = "https://github.com/Madgeniusblink/eth-price-prediction/tree/main/reports/latest"

            # Fallback to root if reports/latest not yet populated
            if not os.path.exists(predictions_file):
                predictions_file = str(script_dir / "predictions_summary.json")
            if not os.path.exists(signals_file):
                signals_file = str(script_dir / "trading_signals.json")

            pred_ok    = os.path.exists(predictions_file)
            signals_ok = os.path.exists(signals_file)
            print(f"  predictions_summary.json : {'✓' if pred_ok else '✗'} ({predictions_file})")
            print(f"  trading_signals.json     : {'✓' if signals_ok else '✗'} ({signals_file})")

            if pred_ok and signals_ok:
                # Save quant_data to a temp file so subprocess can read it
                import tempfile, json as _json
                _slack_env = dict(os.environ)
                if quant_data:
                    _qf_tmp = os.path.join(os.path.dirname(predictions_file), 'quant_data.json')
                    with open(_qf_tmp, 'w') as _f:
                        _json.dump(quant_data, _f, indent=2, default=str)
                result4 = subprocess.run(
                    [sys.executable, "src/send_slack_notification.py",
                     predictions_file, signals_file, report_url],
                    check=True, capture_output=True, text=True, env=_slack_env
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
