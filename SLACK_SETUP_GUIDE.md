# Slack Webhook Setup Guide

## What This Enables

Once configured, your ETH prediction system will automatically send you Slack notifications for:

- **Critical Alerts** - System failures, data issues
- **Warnings** - Performance degradation, stale data
- **Success Notifications** - New predictions with trading signals
- **Health Check Reports** - System status updates

Without this configuration, the system will still run and log everything, but you won't receive real-time notifications.

---

## Step 1: Create a Slack Incoming Webhook

### Option A: If You Already Have a Slack Workspace

1. Go to https://api.slack.com/apps
2. Click **"Create New App"**
3. Select **"From scratch"**
4. Give it a name (e.g., "ETH Prediction Bot")
5. Select your workspace
6. Click **"Create App"**

### Option B: If You Need to Create a Slack Workspace First

1. Go to https://slack.com/create
2. Enter your email and create a new workspace
3. Follow the prompts to set up your workspace
4. Then follow Option A above

---

## Step 2: Enable Incoming Webhooks

1. In your app settings, click **"Incoming Webhooks"** in the left sidebar
2. Toggle the switch to **"On"**
3. Scroll down and click **"Add New Webhook to Workspace"**
4. Select the channel where you want notifications (e.g., #eth-predictions)
5. Click **"Allow"**

You will now see a webhook URL that looks like:
```
https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

**IMPORTANT:** Copy this URL - you'll need it in the next step.

---

## Step 3: Add Webhook to GitHub Secrets

1. Go to your GitHub repository: https://github.com/Madgeniusblink/eth-price-prediction
2. Click **"Settings"** (top menu)
3. In the left sidebar, click **"Secrets and variables"** → **"Actions"**
4. Click **"New repository secret"**
5. Name: `SLACK_WEBHOOK_URL`
6. Value: Paste the webhook URL you copied in Step 2
7. Click **"Add secret"**

---

## Step 4: Verify Configuration

### Test the Webhook Manually

You can test if the webhook works by running this command in your terminal:

```bash
curl -X POST -H 'Content-type: application/json' \
--data '{"text":"Test message from ETH Prediction System"}' \
YOUR_WEBHOOK_URL_HERE
```

Replace `YOUR_WEBHOOK_URL_HERE` with your actual webhook URL.

If successful, you should see "ok" as the response and receive a message in your Slack channel.

### Wait for Next Automated Run

The system runs every 4 hours at:
- 00:00 UTC (7pm EST / 4pm PST)
- 04:00 UTC (11pm EST / 8pm PST)
- 08:00 UTC (3am EST / 12am PST)
- 12:00 UTC (7am EST / 4am PST)
- 16:00 UTC (11am EST / 8am PST)
- 20:00 UTC (3pm EST / 12pm PST)

After the next run, you should receive a Slack notification with:
- Current ETH price
- Predictions for 15m, 30m, 1h, 2h
- Trading signal (BUY/SELL/HOLD/WAIT)
- Confidence level
- Entry, target, and stop-loss prices

---

## What You'll Receive

### Success Notification (Every 4 Hours)
```
New ETH Price Prediction

Current Price: $2,963.90

Predictions:
  • 15m: $2,970.50 (+0.22%)
  • 30m: $2,985.25 (+0.72%)
  • 1h: $3,000.00 (+1.22%)
  • 2h: $3,015.50 (+1.74%)

Trading Signal: BUY
Confidence: HIGH
Entry: $2,965.00
Target: $3,015.00
Stop Loss: $2,940.00

View full report: https://github.com/...
```

### Critical Alert (If System Fails)
```
CRITICAL: System has failed 3 consecutive times

Context:
{
  "last_error": "FileNotFoundError: eth_4h_data.csv"
}
```

### Warning (If Performance Degrades)
```
WARNING: Model directional accuracy is below 50%: 42.9%
```

---

## Troubleshooting

### Not Receiving Notifications?

1. **Check GitHub Actions Logs**
   - Go to your repository → Actions tab
   - Click on the latest workflow run
   - Check if there are any errors in the "Generate prediction report" step

2. **Verify Secret is Set**
   - Go to repository Settings → Secrets and variables → Actions
   - Confirm `SLACK_WEBHOOK_URL` is listed

3. **Test Webhook Directly**
   - Use the curl command from Step 4 to verify the webhook works

4. **Check Slack Channel**
   - Make sure the app has permission to post to your selected channel
   - Check if the app was removed or permissions were revoked

### Webhook URL Changed?

If you need to update the webhook URL:
1. Go to repository Settings → Secrets and variables → Actions
2. Click on `SLACK_WEBHOOK_URL`
3. Click "Update secret"
4. Paste the new URL
5. Click "Update secret"

---

## Optional: Customize Notifications

If you want to customize the notification format or add more channels, you can modify:

**File:** `src/alert_system.py`

**What you can customize:**
- Message format and styling
- Color coding for different alert levels
- Additional channels for different alert types
- Frequency of certain notifications

---

## What Happens Without Slack Configured?

The system will still work perfectly fine:
- All predictions will be generated
- Reports will be committed to GitHub
- Logs will be written to `/logs/`
- Health monitoring will track everything
- You just won't get real-time notifications

You can always check:
- Latest report: `reports/latest/README.md`
- Dashboard: `reports/dashboard.html`
- Logs: `logs/system_YYYYMMDD.log`

---

## Summary

**Required for Slack notifications:**
1. Create Slack app and enable incoming webhooks
2. Add webhook URL to GitHub repository secrets as `SLACK_WEBHOOK_URL`

**That's it!** The system will automatically start sending notifications on the next run.

**No Slack configuration needed for:**
- System to run automatically
- Predictions to be generated
- Reports to be created
- Logs to be written
- Self-healing to work

Slack is purely for real-time notifications to keep you informed without having to check GitHub manually.
