# Automated Prediction and Trading Signal System

This document details the automated system that generates and commits prediction reports and trading signals every 4 hours.

## System Overview

The automation is driven by a GitHub Actions workflow that executes a Python script (`src/generate_report.py`) on a schedule. This script is the master orchestrator for the entire process.

### Workflow Steps

1.  **Scheduled Trigger**: The GitHub Actions workflow is triggered every 4 hours (`cron: '0 */4 * * *'`).
2.  **Environment Setup**: A virtual environment is created, and all dependencies from `requirements.txt` are installed.
3.  **Execute `generate_report.py`**: The main script runs, performing the following sub-steps:
    *   **Data Fetching**: Fetches the latest 1-minute ETH/USD data from the Binance API.
    *   **Prediction Generation**: Runs the ensemble machine learning model to generate price forecasts.
    *   **Trading Signal Analysis**: Executes the `trading_signals` module to determine the market trend, support/resistance levels, and generate a BUY/SELL/SHORT/WAIT signal.
    *   **Visualization**: Creates the three standard PNG charts.
    *   **Report Consolidation**: Gathers all predictions, signals, charts, and analysis into a single, comprehensive `README.md` file.
4.  **File Organization**:
    *   A new, timestamped directory is created under `reports/YYYY/MM/DD/`.
    *   All generated files (JSON, CSV, PNG, MD) are saved to this new directory.
    *   The files are also copied to the `reports/latest/` directory, overwriting the previous report for easy access.
5.  **Commit to GitHub**: The GitHub Action automatically commits all new and updated files back to the repository, creating a permanent, public track record.

## Report Contents

Each automated report is a complete analysis package, containing:

-   **Executive Summary**: High-level overview of the market and the primary trading signal.
-   **Price Predictions**: Forecasts for 15m, 30m, 60m, and 120m horizons.
-   **Trading Analysis**: The core of the report, detailing the market trend, support/resistance levels, and the full trade setup (entry, stop, target).
-   **Prediction Charts**: Visualizations of the predictions and technical indicators.
-   **Model Performance**: R² scores and weights for transparency.
-   **Terminology Guide**: Plain English explanations of all technical terms.
-   **Raw Data Files**: All underlying data in JSON and CSV formats.

## How to Manage the Automation

### Enabling/Disabling the Workflow

1.  Go to the **Actions** tab of the repository on GitHub.
2.  Select the **Scheduled Prediction Report** workflow.
3.  You can manually run the workflow or disable it using the options provided.

### Changing the Schedule

1.  Edit the `.github/workflows/scheduled-prediction.yml` file.
2.  Modify the `cron` schedule. For example, to run every hour, change it to `'0 * * * *'`.

### Local Execution

You can simulate the automated run locally at any time:

```bash
python src/generate_report.py
```

This is useful for testing changes before committing them.

## Folder Structure for Reports

```
reports/
├── latest/                  # Always contains the most recent report
│   ├── README.md
│   ├── eth_1m_data.csv
│   ├── eth_1hour_prediction.png
│   ├── eth_prediction_overview.png
│   ├── eth_technical_indicators.png
│   ├── predictions_summary.json
│   └── trading_signals.json
└── 2026/
    └── 01/
        └── 05/
            ├── 2026-01-05_12-00_README.md
            ├── 2026-01-05_12-00_data.csv
            ├── 2026-01-05_12-00_1hour.png
            ├── ... (and all other files)
            └── 2026-01-05_16-00_README.md
            └── ... (next report)
```

This structure ensures both a clean historical archive and easy access to the latest analysis.
