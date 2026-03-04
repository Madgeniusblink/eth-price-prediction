#!/usr/bin/env python3
"""
Send ETH Quant Report to Slack.
Full institutional format: DeFi LP + SushiSwap + Fear&Greed + Kelly + ensemble predictions.
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone

SIGNAL_EMOJI = {'BUY': '🟢', 'SELL': '🔴', 'SHORT': '🟠', 'WAIT': '⏸️', 'HOLD': '⏸️'}
SIGNAL_COLOR = {'BUY': '#2ecc71', 'SELL': '#e74c3c', 'SHORT': '#e67e22',
                'WAIT': '#95a5a6', 'HOLD': '#95a5a6'}


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load {path}: {e}")
        return {}


def _fmt_pred(p):
    price = p.get('price', 0)
    chg   = p.get('change_pct', p.get('change_percent', 0))
    band  = p.get('confidence_band', 0)
    if not price:
        return 'N/A'
    arrow = '▲' if chg >= 0 else '▼'
    band_str = f" (±${band:.0f})" if band else ""
    return f"${price:,.2f}{band_str} {arrow}{abs(chg):.2f}%"


def build_message(predictions: dict, signals: dict) -> tuple[str, str]:
    """Build full quant-grade Slack message. Returns (text, signal)."""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')

    # ── Price & predictions ───────────────────────────────────────────────────
    current_price = predictions.get('current_price', 0)
    preds = predictions.get('predictions', {})
    p15  = preds.get('15min', preds.get('15m', {}))
    p30  = preds.get('30min', preds.get('30m', {}))
    p60  = preds.get('60min', preds.get('60m', {}))
    p120 = preds.get('120min', preds.get('120m', {}))

    # ── Trading signals ───────────────────────────────────────────────────────
    trading  = signals.get('trading_signal', {})
    trend_a  = signals.get('trend_analysis', {})
    sr       = signals.get('support_resistance', {})
    defi_sig = signals.get('defi_signals', {})

    # Use signals price if closer to live
    signals_price = sr.get('current_price', 0)
    if signals_price and abs(signals_price - current_price) / max(current_price, 1) > 0.05:
        current_price = signals_price

    sig       = trading.get('signal', 'WAIT')
    confidence= trading.get('confidence', 'MEDIUM')
    entry     = trading.get('entry', current_price)
    stop_loss = trading.get('stop_loss', 0)
    target    = trading.get('target', 0)
    trend     = trend_a.get('trend', 'NEUTRAL')
    rsi_val   = trend_a.get('rsi', 0) or 0

    # Signal score & Kelly from trading signals (if computed)
    score_data  = signals.get('signal_score', {})
    score       = score_data.get('score', 50)
    kelly_data  = signals.get('kelly', {})
    kelly_pct   = kelly_data.get('kelly_pct', 5.0)

    # ── On-chain data (stored in predictions_summary.json) ────────────────────
    onchain   = predictions.get('onchain_data') or {}
    fg        = onchain.get('fear_greed', {})
    gas_data  = onchain.get('gas', {})
    uni_pool  = onchain.get('uniswap_v3_pool', {})
    onchain_mom = onchain.get('onchain_momentum', 'NEUTRAL')

    fg_val    = fg.get('value')
    fg_label  = fg.get('label', '')
    gas_gwei  = gas_data.get('fast_gas_gwei')
    gas_label = gas_data.get('label', 'N/A')

    # ── DeFi signals (stored in trading_signals.json under defi_signals) ──────
    uni_lp    = defi_sig.get('uniswap_v3', {})
    proto_cmp = defi_sig.get('protocol_comparison', {})

    tvl_usd       = uni_pool.get('tvl_usd') or uni_lp.get('tvl_usd') or 0
    fees_24h      = uni_pool.get('fees_24h_usd') or uni_lp.get('fees_24h_usd') or 0
    fee_apy       = uni_lp.get('projected_fee_apy')
    range_1s      = uni_lp.get('optimal_range_1sigma', [None, None])
    il_10         = uni_lp.get('il_at_10pct', 0)
    il_20         = uni_lp.get('il_at_20pct', 0)
    lp_reco       = uni_lp.get('recommendation', 'WAIT')
    better_proto  = proto_cmp.get('better_protocol', 'N/A')
    better_apy    = proto_cmp.get('better_apy', 0)

    # ── Backtest data ─────────────────────────────────────────────────────────
    bt_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'reports', 'latest', 'backtest_summary.json')
    bt = {}
    try:
        if os.path.exists(bt_path):
            with open(bt_path) as f:
                bt = json.load(f)
    except Exception:
        pass

    sharpe   = bt.get('sharpe_ratio', 'N/A')
    dir_acc  = bt.get('direction_accuracy_pct', 'N/A')

    sig_emoji = SIGNAL_EMOJI.get(sig, '⏸️')

    # ── Helper fmts ───────────────────────────────────────────────────────────
    def _m(v, fmt='.2f'):
        return format(v, fmt) if isinstance(v, (int, float)) else 'N/A'

    fg_str    = f"{fg_val} ({fg_label})" if fg_val is not None else 'N/A'
    gas_str   = f"{_m(gas_gwei, '.0f')} gwei ({gas_label})" if gas_gwei else 'N/A'
    tvl_m     = f"${tvl_usd/1e6:.1f}M" if tvl_usd else 'N/A'
    fees_k    = f"${fees_24h/1e3:.1f}K" if fees_24h else 'N/A'
    apy_str   = f"{_m(fee_apy)}%" if fee_apy is not None else 'N/A'
    range_str = (f"${range_1s[0]:,.0f} – ${range_1s[1]:,.0f}"
                 if range_1s and range_1s[0] else 'N/A')
    il10_str  = f"{_m(il_10)}%"
    il20_str  = f"{_m(il_20)}%"
    entry_str = f"${entry:,.2f}" if entry else 'N/A'
    stop_str  = f"${stop_loss:,.2f}" if stop_loss else 'N/A'
    tgt_str   = f"${target:,.2f}" if target else 'N/A'
    sharpe_str = f"{sharpe:.4f}" if isinstance(sharpe, float) else str(sharpe)
    acc_str   = f"{dir_acc:.1f}%" if isinstance(dir_acc, float) else str(dir_acc)

    text = (
        f"*ETH Quant Report — {timestamp} UTC*\n"
        f"*Live Price:* ${current_price:,.2f} (CoinGecko)\n"
        f"*Signal Score:* {score}/100 → {sig}\n"
        f"*Confidence:* {confidence}\n"
        f"*Kelly Fraction:* {kelly_pct:.1f}% of position\n"
        f"\n"
        f"📊 *Price Predictions (Ensemble)*\n"
        f"• 15m: {_fmt_pred(p15)}\n"
        f"• 30m: {_fmt_pred(p30)}\n"
        f"• 1h:  {_fmt_pred(p60)}\n"
        f"• 2h:  {_fmt_pred(p120)}\n"
        f"\n"
        f"🦄 *Uniswap v3 ETH/USDC Pool*\n"
        f"• TVL: {tvl_m} | 24h Fees: {fees_k}\n"
        f"• Fee APY: {apy_str}\n"
        f"• Optimal LP Range: {range_str}\n"
        f"• IL at 10%: {il10_str} | IL at 20%: {il20_str}\n"
        f"• Recommendation: {lp_reco}\n"
        f"\n"
        f"🍣 *SushiSwap vs Uniswap*\n"
        f"• Better APY: {better_proto} ({_m(better_apy)}%)\n"
        f"\n"
        f"😱 *Market Sentiment*\n"
        f"• Fear & Greed: {fg_str}\n"
        f"• Gas: {gas_str}\n"
        f"• On-chain signal: {onchain_mom}\n"
        f"\n"
        f"*Entry:* {entry_str} | *Target:* {tgt_str} | *Stop:* {stop_str}\n"
        f"\n"
        f"📉 *Backtest (90d)*: Sharpe {sharpe_str} | Dir.Acc {acc_str}\n"
        f"\n"
        f"_⚠️ Analysis only — not financial advice_"
    )

    return text, sig


def send_slack_notification(predictions_file, signals_file, report_url=""):
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("Error: SLACK_WEBHOOK_URL not set")
        sys.exit(1)

    predictions = _load_json(predictions_file)
    signals     = _load_json(signals_file)

    text, sig = build_message(predictions, signals)
    color      = SIGNAL_COLOR.get(sig, '#95a5a6')

    payload = {
        "attachments": [{"color": color, "text": text, "mrkdwn_in": ["text"]}]
    }

    try:
        r = requests.post(webhook_url, json=payload, timeout=15)
        r.raise_for_status()
        print("✅ Slack notification sent!")
        print("\n── Preview ──")
        print(text)
    except Exception as e:
        print(f"❌ Slack send failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: send_slack_notification.py <predictions_file> <signals_file> [report_url]")
        sys.exit(1)
    send_slack_notification(sys.argv[1], sys.argv[2],
                            sys.argv[3] if len(sys.argv) > 3 else "")
