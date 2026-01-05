# Data Science Methodology

**Document Version:** 1.0  
**Last Updated:** January 5, 2026

---

## 1. Introduction

This document outlines the comprehensive data science methodology underpinning the Ethereum Short-Term Price Prediction System. The primary objective of this system is to forecast the price of Ethereum (ETH) over a 1-2 hour horizon with a high degree of accuracy and reliability. 

Our approach is grounded in the principle that financial time-series data, while stochastic, contains discernible patterns and dependencies. By combining classical statistical methods with modern machine learning techniques, we can model these patterns to generate probabilistic forecasts. This methodology emphasizes transparency, continuous validation, and a multi-faceted approach to modeling to ensure robustness in the dynamic and volatile cryptocurrency market.

## 2. Core Methodological Principles

Our system is built upon a foundation of six core principles that guide every stage of the process, from data acquisition to prediction generation.

| Principle | Description |
| :--- | :--- |
| **1. Data Quality First** | The predictive power of any model is fundamentally limited by the quality of its input data. We prioritize using high-frequency, high-liquidity data from a reputable source (Binance) and implement rigorous validation checks to ensure data integrity. |
| **2. Rich Feature Engineering** | Raw price data alone is insufficient. We engineer a rich set of over 15 features, primarily technical indicators, to provide the models with a multi-dimensional view of market dynamics, including trend, momentum, volatility, and volume. |
| **3. Model Diversity** | No single model is optimal for all market conditions. We employ a diverse set of models—linear, polynomial, and a tree-based ensemble—each designed to capture different types of patterns, from simple linear trends to complex non-linear relationships. |
| **4. Ensemble Learning** | To improve robustness and reduce model-specific errors, we utilize an ensemble approach. Predictions from individual models are combined using a weighted average, where weights are dynamically assigned based on each model's recent, validated performance. |
| **5. Rigorous & Continuous Validation** | A model's past performance is no guarantee of future results. We implement a continuous validation framework using time-series-appropriate techniques like rolling-window backtesting and walk-forward analysis to constantly assess and ensure model reliability. |
| **6. Risk & Uncertainty Quantification** | Price prediction is inherently probabilistic. We do not provide single-point forecasts but rather a range of likely outcomes, quantified through confidence intervals. This provides users with a more realistic understanding of prediction uncertainty. |

## 3. Data Acquisition and Preprocessing

### 3.1. Data Source

The system utilizes the public API of **Binance**, one of the world's largest and most liquid cryptocurrency exchanges. This choice ensures that the data reflects a significant portion of global trading activity and is less susceptible to localized price manipulation.

- **Endpoint**: `https://api.binance.com/api/v3/klines`
- **Asset**: `ETHUSDT` (Ethereum vs. Tether)
- **Granularity**: 1-minute (1m) candlesticks

### 3.2. Data Collection

The `fetch_data.py` script retrieves the most recent 500 1-minute OHLCV (Open, High, Low, Close, Volume) data points. This provides approximately 8.3 hours of historical context, which is a suitable look-back period for a 1-2 hour forecast horizon.

### 3.3. Data Preprocessing and Validation

Upon retrieval, the raw data undergoes several validation and cleaning steps:

1.  **Timestamp Conversion**: Timestamps are converted from Unix milliseconds to a standard datetime format.
2.  **Data Type Coercion**: All price and volume columns are converted to floating-point numbers to ensure compatibility with numerical algorithms.
3.  **Missing Value Check**: The system checks for any gaps in the time series. Given the high liquidity of the ETHUSDT pair on Binance, missing values are rare. If detected, a simple forward-fill strategy would be employed, but the primary response is to flag the data as potentially unreliable.
4.  **Outlier Detection**: Extreme price changes (e.g., >10% in one minute) are flagged. While these are often legitimate in crypto markets, they are noted as they can disproportionately influence model training.

## 4. Feature Engineering

Feature engineering is the most critical step in translating raw price data into meaningful inputs for our models. The `technical_indicators.py` module calculates a suite of features designed to capture different facets of market behavior.

| Feature Category | Indicators | Purpose |
| :--- | :--- | :--- |
| **Trend** | SMA (5, 10, 20, 50), EMA (5, 10, 20) | Identify the underlying direction and strength of the price trend across multiple timeframes. |
| **Momentum** | RSI (14), MACD (12, 26, 9), Momentum (10) | Measure the velocity and rate of change of price movements, helping to identify overbought/oversold conditions. |
| **Volatility** | Bollinger Bands (20, 2), Standard Deviation (20) | Quantify the magnitude of price fluctuations, indicating market risk and potential for breakouts. |
| **Volume** | Volume SMA (20), Volume Ratio | Analyze trading activity to confirm the strength of a trend. High volume on a price move is a sign of conviction. |

These features are calculated on a rolling basis and are combined into a single feature matrix that serves as the input for the machine learning model.

## 5. Modeling Approach

Our system employs a three-tiered modeling strategy, culminating in a performance-weighted ensemble forecast.

### 5.1. Model 1: Linear Regression

- **Purpose**: To establish a baseline trend.
- **Methodology**: A simple linear regression model is fitted to the last 100 closing prices against a time index. It captures the primary linear trend in the recent data. While simplistic, it provides a stable, non-volatile anchor for the ensemble.

### 5.2. Model 2: Polynomial Regression

- **Purpose**: To capture non-linear, curve-like patterns.
- **Methodology**: A polynomial regression model (degree 2) is fitted to the same 100-period window. This model can capture acceleration and deceleration in price, making it effective at modeling short-term parabolic moves or reversals that a linear model would miss.

### 5.3. Model 3: Random Forest Regressor

- **Purpose**: To model complex, non-linear interactions between features.
- **Methodology**: A Random Forest model, an ensemble of 100 decision trees, is trained on the rich feature matrix. Each tree learns a different set of rules based on the technical indicators. This model is the powerhouse of the system, capable of identifying complex conditions like "*if RSI is below 30 and the price is crossing above the lower Bollinger Band, then the price is likely to increase.*"

### 5.4. Ensemble Method

- **Purpose**: To create a robust forecast by combining the strengths of each model.
- **Methodology**: The final prediction is a weighted average of the outputs from the three models. The weights are not static; they are calculated dynamically based on the recent R² performance of each model, as determined by the continuous validation process. This ensures that models performing well in the current market conditions are given a higher influence on the final forecast.

    **Formula**: `Ensemble_Pred = (w_L * Pred_L) + (w_P * Pred_P) + (w_RF * Pred_RF)`
    where `w` is the performance-based weight and `Pred` is the prediction for each model (Linear, Polynomial, Random Forest).

## 6. Prediction Generation

1.  **Iterative Forecasting**: For the Random Forest model, predictions are made one step at a time. The prediction for time `t+1` is used to update the input features to predict for `t+2`, and so on. This iterative process generates a full forecast path.
2.  **Confidence Interval Estimation**: To quantify uncertainty, a simplified confidence interval is calculated around the ensemble prediction. The width of this interval is based on the standard deviation of the historical price data over the last 20 periods. A wider band indicates higher recent volatility and thus greater prediction uncertainty.
3.  **Output Generation**: The system outputs specific price predictions for key time horizons (15, 30, 60, 120 minutes) and saves the full predicted path for visualization.

## 7. Conclusion

This methodology provides a robust and transparent framework for short-term cryptocurrency price prediction. By combining a rigorous data-driven approach with a diverse set of modeling techniques and a commitment to continuous validation, the system is designed to adapt to the ever-changing dynamics of the market. The emphasis on feature engineering, ensemble learning, and uncertainty quantification ensures that the resulting predictions are not just a single number, but a well-reasoned and contextualized forecast.


## 8. Trading Signal Methodology

While price prediction provides a forecast of *what* the price might do, the trading signal component is designed to provide actionable guidance on *when* to act. This is achieved through a two-stage process that prioritizes market context over raw signals, a crucial step in avoiding common trading pitfalls.

### 8.1. Stage 1: Trend Detection

Before generating any entry or exit signal, the system first establishes the broader market context. It classifies the market into one of three regimes:

-   **BULL MARKET**: A sustained uptrend, identified by the 50-period moving average trading above the 200-period moving average (a "Golden Cross") and a consistent pattern of higher-lows in the price action.
-   **BEAR MARKET**: A sustained downtrend, identified by the 50-period moving average trading below the 200-period moving average (a "Death Cross") and a pattern of lower-highs.
-   **NEUTRAL / RANGING**: A sideways market where trend indicators are mixed and the price is oscillating between clear support and resistance levels.

This initial classification is the most critical step, as it dictates the types of trades the system will look for. In a Bull Market, the system will only look for BUY signals on dips. In a Bear Market, it will only look for SHORT signals on rallies.

### 8.2. Stage 2: Signal Generation

Once the market trend is established, the system seeks a high-probability entry point by requiring a confluence of evidence from multiple indicators. A signal is only generated when several conditions are met simultaneously:

| Signal | Required Trend | Price Location | Momentum Conditions |
| :--- | :--- | :--- | :--- |
| **BUY** | BULL MARKET | Near a key **Support** level | RSI is oversold or neutral, and a bullish MACD crossover is forming. |
| **SHORT** | BEAR MARKET | Near a key **Resistance** level | RSI is overbought or neutral, and a bearish MACD crossover is forming. |
| **SELL** | Any | Near a key **Resistance** level | RSI is overbought, indicating a potential exit point for a long position. |
| **WAIT** | Any | Not near key levels | Indicators are conflicting, providing no clear, high-probability setup. |

### 8.3. Risk Management Integration

For every BUY or SHORT signal generated, the system also calculates a complete trade setup:

-   **Entry Price**: The current price at the time of the signal.
-   **Stop Loss**: Placed just below the identified support level (for a BUY) or just above the resistance level (for a SHORT). This defines the maximum acceptable loss for the trade.
-   **Target Price**: The next significant resistance level (for a BUY) or support level (for a SHORT).
-   **Risk/Reward Ratio**: The ratio of the potential profit (Target - Entry) to the potential loss (Entry - Stop Loss). The system is designed to favor trades with a high Risk/Reward Ratio (typically > 1.5).

This comprehensive approach ensures that every signal is not just a prediction, but a complete, risk-managed trade idea that is aligned with the broader market trend.
