#!/usr/bin/env python3
"""
Send formatted Slack notification with ETH predictions, signal, and derivatives context.
Single clean message — no duplicate alerts.
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone


SIGNAL_EMOJI = {
    'BUY':   '🟢',
    'SELL':  '🔴',
    'SHORT': '🟠',
    'WAIT':  '⏸️',
    'HOLD':  '⏸️',
}

SIGNAL_COLOR = {
    'BUY':   '#2ecc71',
    'SELL':  '#e74c3c',
    'SHORT': '#e67e22',
    'WAIT':  '#95a5a6',
    'HOLD':  '#95a5a6',
}


def send_slack_notification(predictions_file, signals_file, report_url):
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("Error: SLACK_WEBHOOK_URL not set")
        sys.exit(1)

    try:
        with open(predictions_file) as f:
            predictions = json.load(f)
        with open(signals_file) as f:
            signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data files: {e}")
        sys.exit(1)

    # ── Core data ───────────────────────────────────────────────────────────
    current_price = predictions.get('current_price', 0)
    preds         = predictions.get('predictions', {})
    p15  = preds.get('15m',  preds.get('15min',  {}))
    p30  = preds.get('30m',  preds.get('30min',  {}))
    p60  = preds.get('60m',  preds.get('60min',  {}))
    p120 = preds.get('120m', preds.get('120min', {}))

    trading   = signals.get('trading_signal', {})
    trend_a   = signals.get('trend_analysis', {})
    sr        = signals.get('support_resistance', {})
    deriv     = signals.get('derivatives_context', {})
    mkt_filt  = signals.get('market_filters', {})

    signal     = trading.get('signal', 'WAIT')
    action     = trading.get('action', 'Monitor position')
    confidence = trading.get('confidence', 'MEDIUM')
    entry      = trading.get('entry', current_price)
    stop_loss  = trading.get('stop_loss', 0)
    target     = trading.get('target', 0)
    rr         = trading.get('risk_reward', 0)
    reasoning  = trading.get('reasoning', '')

    trend      = trend_a.get('trend', 'NEUTRAL')
    rsi_val    = trend_a.get('rsi', 0)
    macd_str   = str(trend_a.get('macd_signal', 'neutral')).upper()
    support    = sr.get('nearest_support', 0)
    resistance = sr.get('nearest_resistance', 0)

    # ── Derivatives context ─────────────────────────────────────────────────
    funding_rate = deriv.get('funding_rate')
    fg_value     = deriv.get('fear_greed_index')
    fg_label     = deriv.get('fear_greed_classification', '')
    lsr          = deriv.get('long_short_ratio')
    oi           = deriv.get('open_interest')

    funding_str = f"{funding_rate*100:.4f}%" if funding_rate is not None else "N/A"
    fg_str      = f"{fg_value} ({fg_label})" if fg_value is not None else "N/A"
    lsr_str     = f"{lsr:.2f}" if lsr is not None else "N/A"
    oi_str      = f"${oi/1e9:.2f}B" if oi is not None else "N/A"

    # ── Load accuracy stats if available ────────────────────────────────────
    acc_str = "Building… (<30 validated)"
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        acc_file = os.path.join(base_dir, 'data', 'accuracy_stats.json')
        if os.path.exists(acc_file):
            with open(acc_file) as f:
                acc = json.load(f)
            wr1 = acc.get('win_rate_1h')
            wr2 = acc.get('win_rate_2h')
            total = acc.get('total_signals', 0)
            if wr1 is not None:
                acc_str = f"1h: {wr1:.0f}% | 2h: {wr2:.0f}% ({total} signals)"
    except Exception:
        pass

    # ── Format ──────────────────────────────────────────────────────────────
    sig_emoji = SIGNAL_EMOJI.get(signal, '⏸️')
    color     = SIGNAL_COLOR.get(signal, '#95a5a6')
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    # Prediction line
    def fmt_pred(p):
        price  = p.get('price', 0)
        chg    = p.get('change_pct', p.get('change_percent', 0))
        arrow  = '▲' if chg >= 0 else '▼'
        return f"${price:,.2f} ({arrow}{abs(chg):.2f}%)"

    rr_display = f"{rr:.2f}:1" if rr and rr > 0 else "—"

    # Trade setup line (only show if actionable signal)
    trade_setup = ""
    if signal in ('BUY', 'SHORT', 'SELL') and entry and stop_loss and target:
        trade_setup = (
            f"*Entry:* ${entry:,.2f}  |  *Stop:* ${stop_loss:,.2f}  "
            f"|  *Target:* ${target:,.2f}  |  *R:R:* {rr_display}"
        )

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"ETH Prediction · {timestamp}"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Price*\n${current_price:,.2f}"},
                {"type": "mrkdwn", "text": f"*Trend*\n{trend}"},
                {"type": "mrkdwn", "text": f"*Signal*\n{sig_emoji} {signal} ({confidence})"},
                {"type": "mrkdwn", "text": f"*Action*\n{action}"},
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Price Targets*"},
            "fields": [
                {"type": "mrkdwn", "text": f"*15m*\n{fmt_pred(p15)}"},
                {"type": "mrkdwn", "text": f"*30m*\n{fmt_pred(p30)}"},
                {"type": "mrkdwn", "text": f"*1h*\n{fmt_pred(p60)}"},
                {"type": "mrkdwn", "text": f"*2h*\n{fmt_pred(p120)}"},
            ]
        },
    ]

    # Trade setup block (only if actionable)
    if trade_setup:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Trade Setup*\n{trade_setup}"}
        })

    blocks += [
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Technical & Derivatives*"},
            "fields": [
                {"type": "mrkdwn", "text": f"*RSI*\n{rsi_val:.1f}"},
                {"type": "mrkdwn", "text": f"*MACD*\n{macd_str}"},
                {"type": "mrkdwn", "text": f"*Support*\n${support:,.2f}"},
                {"type": "mrkdwn", "text": f"*Resistance*\n${resistance:,.2f}"},
                {"type": "mrkdwn", "text": f"*Funding Rate*\n{funding_str}"},
                {"type": "mrkdwn", "text": f"*Fear & Greed*\n{fg_str}"},
                {"type": "mrkdwn", "text": f"*L/S Ratio*\n{lsr_str}"},
                {"type": "mrkdwn", "text": f"*Open Interest*\n{oi_str}"},
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Signal Win Rate*\n{acc_str}"},
                {"type": "mrkdwn", "text": f"*<{report_url}|View Report>*\nGitHub"},
            ]
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn",
                 "text": "⚠️ _Analysis only — not financial advice. Trade at your own risk._"}
            ]
        },
    ]

    # Charts
    repo_raw = "https://raw.githubusercontent.com/Madgeniusblink/eth-price-prediction/main/reports/latest"
    for img_url, alt in [
        (f"{repo_raw}/eth_predictions_overview.png", "Overview"),
        (f"{repo_raw}/eth_2hour_prediction.png",     "2h Prediction"),
    ]:
        blocks.append({"type": "image", "image_url": img_url, "alt_text": alt})

    payload = {
        "text": f"ETH {sig_emoji} {signal} @ ${current_price:,.2f} | {trend} | {timestamp}",
        "attachments": [{"color": color, "blocks": blocks}]
    }

    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
        r.raise_for_status()
        print("✓ Slack notification sent successfully")
    except requests.RequestException as e:
        print(f"Error sending Slack notification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: send_slack_notification.py <predictions_file> <signals_file> <report_url>")
        sys.exit(1)
    send_slack_notification(sys.argv[1], sys.argv[2], sys.argv[3])
