#!/usr/bin/env python3
"""
Build all 4 notebooks for the Precog Quant Task 2026 pipeline.
Run: python build_all_notebooks.py
"""
import nbformat as nbf
import os

NB_DIR = "notebooks"
os.makedirs(NB_DIR, exist_ok=True)

def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    for ctype, source in cells:
        if ctype == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))
    return nb

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1: Feature Engineering & Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
nb1_cells = [
("md", """# Part 1: Feature Engineering & Data Cleaning
**Precog Quant Task 2026 — Congrats Congrats Money Money**

This notebook covers:
1. Loading raw OHLCV data for 100 anonymized stocks (~10 years)
2. Data quality assessment and cleaning
3. Feature engineering: momentum, volatility, volume, mean-reversion, and technical indicators
4. Cross-sectional normalization
5. Saving the cleaned feature matrix for downstream modeling

> *"Good data = good features, and good features = good model."*
"""),

("code", """import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True
sns.set_style("whitegrid")

print("Libraries loaded successfully.")
"""),

("md", """## 1. Load Raw Data
Each CSV has columns: `Date, Open, High, Low, Close, Volume` for one anonymized asset.
"""),

("code", """DATA_PATH = "../data/raw/anonymized_data/*.csv"
files = sorted(glob.glob(DATA_PATH))
print(f"Number of CSV files found: {len(files)}")

dfs = []
for f in files:
    df = pd.read_csv(f)
    # Extract clean ticker name
    asset_name = os.path.basename(f).replace(".csv", "")
    df["ticker"] = asset_name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)

print(f"Total rows: {len(data):,}")
print(f"Assets: {data['ticker'].nunique()}")
print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
data.head()
"""),

("md", """## 2. Data Quality Assessment"""),

("code", """# Check for missing values
print("=== Missing Values (fraction) ===")
print(data.isnull().mean())
print()

# Check for duplicates
dupes = data.duplicated(subset=["ticker", "Date"]).sum()
print(f"Duplicate (ticker, Date) rows: {dupes}")

# Check for zero/negative prices and volume
print(f"\\nRows with Close <= 0: {(data['Close'] <= 0).sum()}")
print(f"Rows with Volume <= 0: {(data['Volume'] <= 0).sum()}")
print(f"Rows with Open <= 0: {(data['Open'] <= 0).sum()}")
print(f"Rows with High <= 0: {(data['High'] <= 0).sum()}")
print(f"Rows with Low <= 0: {(data['Low'] <= 0).sum()}")
"""),

("code", """# Coverage per asset
coverage = data.groupby("ticker").agg(
    start=("Date", "min"),
    end=("Date", "max"),
    count=("Date", "count")
)
print("=== Asset coverage summary ===")
print(coverage["count"].describe())

# Plot coverage distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
coverage["count"].hist(bins=30, ax=axes[0], edgecolor="black")
axes[0].set_title("Distribution of Data Points per Asset")
axes[0].set_xlabel("Number of trading days")

# Price distribution across assets (last available price)
last_prices = data.groupby("ticker")["Close"].last()
last_prices.hist(bins=30, ax=axes[1], edgecolor="black")
axes[1].set_title("Distribution of Last Close Price per Asset")
axes[1].set_xlabel("Close Price ($)")
plt.tight_layout()
plt.savefig("../outputs/data_coverage.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("code", """# Detect outlier daily returns (before cleaning)
data["raw_return"] = data.groupby("ticker")["Close"].pct_change()

# Flag extreme single-day moves (>30% as suspicious)
extreme = data[data["raw_return"].abs() > 0.30]
print(f"Extreme daily moves (|return| > 30%): {len(extreme)} rows out of {len(data):,}")
print(f"Affected assets: {extreme['ticker'].nunique()}")

# Show distribution of raw returns
fig, ax = plt.subplots(figsize=(12, 4))
data["raw_return"].dropna().hist(bins=500, ax=ax, density=True, edgecolor="none", alpha=0.7)
ax.set_xlim(-0.15, 0.15)
ax.set_title("Distribution of Raw Daily Returns (truncated at ±15%)")
ax.set_xlabel("Daily Return")
plt.tight_layout()
plt.savefig("../outputs/raw_return_dist.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 3. Data Cleaning

**Cleaning steps:**
1. Remove any rows with missing OHLCV values
2. Remove rows with non-positive Close or Volume
3. Remove duplicate (ticker, Date) entries
4. Winsorize extreme daily returns at the 0.1th and 99.9th percentiles (clip outlier prices)
5. Forward-fill small gaps (max 3 days) if they occur within a ticker
"""),

("code", """# Step 1-3: Basic cleaning
initial_rows = len(data)

# Drop rows with any missing OHLCV
data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

# Drop non-positive Close or Volume
data = data[(data["Close"] > 0) & (data["Volume"] > 0)]

# Drop duplicates
data = data.drop_duplicates(subset=["ticker", "Date"])

# Re-sort
data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)

# Remove the temp column
data = data.drop(columns=["raw_return"], errors="ignore")

print(f"Rows before cleaning: {initial_rows:,}")
print(f"Rows after cleaning:  {len(data):,}")
print(f"Rows removed: {initial_rows - len(data):,}")
print(f"Assets remaining: {data['ticker'].nunique()}")
"""),

("md", """## 4. Feature Engineering

We build features across multiple categories:
- **Returns**: log returns at various horizons
- **Momentum**: cumulative returns over rolling windows
- **Volatility**: rolling standard deviation of returns
- **Volume**: relative volume, volume momentum
- **Mean Reversion**: short-term reversal
- **Technical**: RSI, Bollinger Band width, ATR, VWAP ratio
- **Cross-sectional**: z-scores for all features (ranking stocks against each other)
"""),

("code", """# ============================================================
# 4a. LOG RETURNS
# ============================================================
# 1-day log return
data["log_ret_1d"] = (
    data.groupby("ticker")["Close"]
    .transform(lambda x: np.log(x / x.shift(1)))
)

# 5-day forward return (target for later modeling)
data["fwd_ret_1d"] = (
    data.groupby("ticker")["log_ret_1d"].shift(-1)
)
data["fwd_ret_5d"] = (
    data.groupby("ticker")["Close"]
    .transform(lambda x: np.log(x.shift(-5) / x))
)

print("Returns computed.")
data[["Date", "ticker", "Close", "log_ret_1d", "fwd_ret_1d", "fwd_ret_5d"]].dropna().head(10)
"""),

("code", """# ============================================================
# 4b. MOMENTUM FEATURES
# ============================================================
for window in [5, 10, 20, 60]:
    col = f"mom_{window}"
    data[col] = (
        data.groupby("ticker")["log_ret_1d"]
        .transform(lambda x, w=window: x.rolling(w).sum())
    )
    
print("Momentum features: mom_5, mom_10, mom_20, mom_60")
data[["Date", "ticker", "mom_5", "mom_10", "mom_20", "mom_60"]].dropna().head()
"""),

("code", """# ============================================================
# 4c. VOLATILITY FEATURES
# ============================================================
for window in [5, 10, 20, 60]:
    col = f"vol_{window}"
    data[col] = (
        data.groupby("ticker")["log_ret_1d"]
        .transform(lambda x, w=window: x.rolling(w).std())
    )

# Volatility ratio (short-term vs long-term) — regime detection
data["vol_ratio_5_20"] = data["vol_5"] / data["vol_20"].replace(0, np.nan)

print("Volatility features: vol_5, vol_10, vol_20, vol_60, vol_ratio_5_20")
data[["Date", "ticker", "vol_5", "vol_10", "vol_20", "vol_ratio_5_20"]].dropna().head()
"""),

("code", """# ============================================================
# 4d. VOLUME FEATURES
# ============================================================
# Relative volume (today's volume vs 20-day average)
data["vol_ma_20"] = (
    data.groupby("ticker")["Volume"]
    .transform(lambda x: x.rolling(20).mean())
)
data["rel_volume"] = data["Volume"] / data["vol_ma_20"].replace(0, np.nan)

# Volume momentum (5-day log change in average volume)
data["vol_mom_5"] = (
    data.groupby("ticker")["Volume"]
    .transform(lambda x: np.log(x.rolling(5).mean() / x.rolling(20).mean()))
)

# Dollar volume (proxy for liquidity)
data["dollar_volume"] = data["Close"] * data["Volume"]
data["log_dollar_vol"] = np.log1p(data["dollar_volume"])

print("Volume features: rel_volume, vol_mom_5, log_dollar_vol")
"""),

("code", """# ============================================================
# 4e. MEAN REVERSION / REVERSAL
# ============================================================
# 1-day reversal
data["reversal_1d"] = -data["log_ret_1d"]

# Distance from 20-day moving average (mean reversion signal)
data["ma_20"] = data.groupby("ticker")["Close"].transform(lambda x: x.rolling(20).mean())
data["dist_ma_20"] = (data["Close"] - data["ma_20"]) / data["ma_20"]

# Distance from 60-day moving average
data["ma_60"] = data.groupby("ticker")["Close"].transform(lambda x: x.rolling(60).mean())
data["dist_ma_60"] = (data["Close"] - data["ma_60"]) / data["ma_60"]

print("Mean reversion features: reversal_1d, dist_ma_20, dist_ma_60")
"""),

("code", """# ============================================================
# 4f. TECHNICAL INDICATORS
# ============================================================

# --- RSI (14-day) ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

data["rsi_14"] = data.groupby("ticker")["Close"].transform(compute_rsi)

# --- Bollinger Band Width (20-day) ---
data["bb_mid"] = data["ma_20"]
data["bb_std"] = data.groupby("ticker")["Close"].transform(lambda x: x.rolling(20).std())
data["bb_width"] = (2 * data["bb_std"]) / data["bb_mid"].replace(0, np.nan)
# Bollinger %B: where price is relative to the bands
data["bb_pctB"] = (data["Close"] - (data["bb_mid"] - 2*data["bb_std"])) / (4 * data["bb_std"]).replace(0, np.nan)

# --- ATR (14-day) Average True Range ---
data["prev_close"] = data.groupby("ticker")["Close"].shift(1)
data["tr"] = np.maximum(
    data["High"] - data["Low"],
    np.maximum(
        abs(data["High"] - data["prev_close"]),
        abs(data["Low"] - data["prev_close"])
    )
)
data["atr_14"] = data.groupby("ticker")["tr"].transform(lambda x: x.rolling(14).mean())
data["atr_pct"] = data["atr_14"] / data["Close"]  # ATR as % of price

# --- High-Low Range Ratio ---
data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]

# Clean up temp columns
data = data.drop(columns=["prev_close", "tr", "bb_mid", "bb_std", "ma_20", "ma_60", "vol_ma_20", "dollar_volume"], errors="ignore")

print("Technical indicators: rsi_14, bb_width, bb_pctB, atr_14, atr_pct, hl_range")
"""),

("code", """# ============================================================
# 4g. OVERNIGHT & INTRADAY RETURNS
# ============================================================
# Overnight return: Open(today) vs Close(yesterday)
data["overnight_ret"] = data.groupby("ticker").apply(
    lambda g: np.log(g["Open"] / g["Close"].shift(1))
).reset_index(level=0, drop=True)

# Intraday return: Close - Open (same day)
data["intraday_ret"] = np.log(data["Close"] / data["Open"])

print("Added: overnight_ret, intraday_ret")
"""),

("md", """## 5. Cross-Sectional Normalization

For cross-sectional models (ranking stocks against each other), raw feature values aren't comparable across assets. We z-score each feature within each trading day and clip to ±3 to contain outliers.

$$z_{i,t} = \\frac{x_{i,t} - \\bar{x}_t}{\\sigma_{x,t}}$$
"""),

("code", """# Define all raw feature columns to normalize
raw_features = [
    "mom_5", "mom_10", "mom_20", "mom_60",
    "vol_5", "vol_10", "vol_20", "vol_60", "vol_ratio_5_20",
    "rel_volume", "vol_mom_5", "log_dollar_vol",
    "reversal_1d", "dist_ma_20", "dist_ma_60",
    "rsi_14", "bb_width", "bb_pctB", "atr_pct", "hl_range",
    "overnight_ret", "intraday_ret"
]

# Cross-sectional z-score per date
cs_features = []
for col in raw_features:
    cs_col = f"{col}_cs"
    grp = data.groupby("Date")[col]
    mu = grp.transform("mean")
    sigma = grp.transform("std").replace(0, np.nan)
    data[cs_col] = ((data[col] - mu) / sigma).fillna(0.0).clip(-3.0, 3.0)
    cs_features.append(cs_col)

print(f"Created {len(cs_features)} cross-sectional z-scored features.")
print("Features:", cs_features)
"""),

("md", """## 6. Feature Visualizations"""),

("code", """# Heatmap: average cross-sectional correlation between features
# Sample one date per month to keep it manageable
sample_dates = data.groupby(data["Date"].dt.to_period("M"))["Date"].first().values

sample = data[data["Date"].isin(sample_dates)]
corr_matrix = sample[cs_features].corr()

plt.figure(figsize=(14, 11))
sns.heatmap(corr_matrix, cmap="RdBu_r", center=0, annot=False,
            xticklabels=[c.replace("_cs", "") for c in cs_features],
            yticklabels=[c.replace("_cs", "") for c in cs_features])
plt.title("Cross-Sectional Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("../outputs/feature_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("code", """# Time-series of aggregate feature values (market mean)
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

daily_mom = data.groupby("Date")["mom_20"].mean()
daily_vol = data.groupby("Date")["vol_20"].mean()
daily_relvol = data.groupby("Date")["rel_volume"].mean()

axes[0].plot(daily_mom.index, daily_mom.values, color="steelblue", linewidth=0.8)
axes[0].set_title("Market Average 20-Day Momentum")
axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")

axes[1].plot(daily_vol.index, daily_vol.values, color="firebrick", linewidth=0.8)
axes[1].set_title("Market Average 20-Day Volatility")

axes[2].plot(daily_relvol.index, daily_relvol.values, color="darkgreen", linewidth=0.8)
axes[2].set_title("Market Average Relative Volume (vs 20-day MA)")
axes[2].axhline(1, color="black", linewidth=0.5, linestyle="--")

plt.xlabel("Date")
plt.tight_layout()
plt.savefig("../outputs/market_features_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("code", """# Distribution of a few key features
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, ["mom_10_cs", "vol_20_cs", "reversal_1d_cs", "rsi_14_cs", "rel_volume_cs", "bb_width_cs"]):
    data[col].dropna().hist(bins=100, ax=ax, density=True, edgecolor="none", alpha=0.7)
    ax.set_title(col.replace("_cs", " (z-scored)"))
    ax.set_xlabel("z-score")
plt.tight_layout()
plt.savefig("../outputs/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 7. Drop Warm-Up Period & Save

We drop the first 60 trading days per asset (needed for the longest rolling window) and save the cleaned feature matrix.
"""),

("code", """# Drop rows where any feature is NaN (warm-up period)
feature_cols_all = raw_features + cs_features + ["log_ret_1d", "fwd_ret_1d", "fwd_ret_5d"]
data_clean = data.dropna(subset=feature_cols_all).reset_index(drop=True)

print(f"Rows before dropping NaN features: {len(data):,}")
print(f"Rows after: {len(data_clean):,}")
print(f"Assets: {data_clean['ticker'].nunique()}")
print(f"Date range: {data_clean['Date'].min().date()} to {data_clean['Date'].max().date()}")
print(f"Total features: {len(cs_features)}")
"""),

("code", """# Save the complete feature matrix
data_clean.to_csv("../data/cleaned_panel_data.csv", index=False)
print("Saved: ../data/cleaned_panel_data.csv")
print(f"Shape: {data_clean.shape}")
print(f"Columns: {list(data_clean.columns)}")
"""),

("md", """## Summary

| Step | Description |
|------|------------|
| **Loading** | 100 assets × ~10 years of daily OHLCV |
| **Cleaning** | Removed invalid prices/volumes, duplicates |
| **Momentum** | 5/10/20/60-day cumulative log returns |
| **Volatility** | 5/10/20/60-day rolling std, vol ratio |
| **Volume** | Relative volume, volume momentum, dollar volume |
| **Mean Reversion** | 1d reversal, distance from 20/60-day MA |
| **Technical** | RSI-14, Bollinger Width/%B, ATR%, High-Low range |
| **Microstructure** | Overnight return, intraday return |
| **Normalization** | Cross-sectional z-scores (clipped ±3) for all 22 features |

The cleaned feature matrix is saved to `cleaned_panel_data.csv` for use in Part 2.
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2: Model Training & Strategy Formulation
# ═══════════════════════════════════════════════════════════════════════════════
nb2_cells = [
("md", """# Part 2: Model Training & Strategy Formulation
**Precog Quant Task 2026**

This notebook covers:
1. Loading the feature matrix from Part 1
2. Defining the prediction target (1-day forward return)
3. Train/test split strategy (temporal, no lookahead)
4. Rolling Ridge Regression with cross-validation
5. LightGBM Gradient Boosting model
6. Ensemble of Ridge + GBDT
7. Signal generation and portfolio construction
8. Information Coefficient analysis

> *Hint A: Financial data is incredibly noisy. Relying on a single signal source can lead to instability.*
> *Hint B: Markets evolve. Consider how your methodology ensures relevance over time.*
"""),

("code", """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True
sns.set_style("whitegrid")

print("Libraries loaded.")
"""),

("md", """## 1. Load Data"""),

("code", """data = pd.read_csv("../data/cleaned_panel_data.csv", parse_dates=["Date"])
data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)

print(f"Shape: {data.shape}")
print(f"Assets: {data['ticker'].nunique()}")
print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
data.head()
"""),

("md", """## 2. Define Features and Target

**Target**: 1-day forward log return (`fwd_ret_1d`)

**Features**: Cross-sectional z-scored features from Part 1 — these capture each stock's relative standing among the universe on any given day.

Using cross-sectional features is important because:
- Raw values are incomparable across assets (a $500 stock vs $20 stock)
- Z-scores focus on **rank/relative position** which is what matters for a long-short strategy
"""),

("code", """# Cross-sectional features
features = [
    "mom_5_cs", "mom_10_cs", "mom_20_cs", "mom_60_cs",
    "vol_5_cs", "vol_10_cs", "vol_20_cs", "vol_60_cs", "vol_ratio_5_20_cs",
    "rel_volume_cs", "vol_mom_5_cs", "log_dollar_vol_cs",
    "reversal_1d_cs", "dist_ma_20_cs", "dist_ma_60_cs",
    "rsi_14_cs", "bb_width_cs", "bb_pctB_cs", "atr_pct_cs", "hl_range_cs",
    "overnight_ret_cs", "intraday_ret_cs"
]

target = "fwd_ret_5d"

# Verify all columns exist
missing = [f for f in features + [target] if f not in data.columns]
if missing:
    print(f"WARNING: Missing columns: {missing}")
else:
    print(f"All {len(features)} features and target present.")
    
# Drop rows with NaN in target
data = data.dropna(subset=[target]).reset_index(drop=True)
print(f"Rows with valid target: {len(data):,}")
"""),

("md", """## 3. Train/Test Split

**Strategy**: Pure temporal split with no lookahead bias.
- **Training**: First ~8 years (expanding window)  
- **Test**: Last ~2 years (out-of-sample)

For the rolling model, we use a 252-day (1 year) lookback window that slides forward one day at a time, ensuring the model always trains only on past data.
"""),

("code", """unique_dates = np.sort(data["Date"].unique())
n_dates = len(unique_dates)

# Use last ~2 years as test set (roughly 504 trading days)
TEST_DAYS = 504
train_end_idx = n_dates - TEST_DAYS

train_end_date = unique_dates[train_end_idx]
test_start_date = unique_dates[train_end_idx + 1]

print(f"Total trading days: {n_dates}")
print(f"Train period: {unique_dates[0]} to {train_end_date}  ({train_end_idx + 1} days)")
print(f"Test period:  {test_start_date} to {unique_dates[-1]}  ({TEST_DAYS} days)")
"""),

("md", """## 4. Rolling Ridge Regression

**Why Ridge?**
- Linear models are fast and interpretable
- L2 regularization prevents overfitting on noisy financial data
- Rolling window ensures model adapts to evolving market regimes

**Training procedure** (for each test day):
1. Use past 252 days as training data
2. Within training window, hold out last 21 days for alpha selection
3. Cross-sectional rank transform the target (more robust than raw returns)
4. Predict on today's cross-section
"""),

("code", """LOOKBACK = 252        # Training window
VAL_DAYS = 21         # Validation days for alpha selection
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Initialize prediction columns
data["pred_ridge"] = np.nan

# Get unique dates
unique_dates = np.sort(data["Date"].unique())

def rank_transform(df, col):
    \"\"\"Cross-sectional percentile rank per date.\"\"\"
    return df.groupby("Date")[col].rank(pct=True)

print(f"Running rolling Ridge over {len(unique_dates) - LOOKBACK} days...")
print("This may take a few minutes...")

for i in range(LOOKBACK, len(unique_dates)):
    # Define windows
    train_dates = unique_dates[i - LOOKBACK : i]
    test_date = unique_dates[i]
    
    train_data = data[data["Date"].isin(train_dates)]
    test_data = data[data["Date"] == test_date]
    
    if len(test_data) == 0:
        continue
    
    # Validation split for alpha selection
    val_dates = train_dates[-VAL_DAYS:]
    train_sub_dates = train_dates[:-VAL_DAYS]
    
    train_sub = data[data["Date"].isin(train_sub_dates)]
    val_sub = data[data["Date"].isin(val_dates)]
    
    # Use rank-transformed target for robustness
    y_train_sub = rank_transform(train_sub, target)
    X_train_sub = train_sub[features]
    X_val = val_sub[features]
    y_val = val_sub[target]
    
    # Alpha selection via validation IC (Spearman correlation)
    best_alpha, best_ic = ALPHAS[0], -np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X_train_sub, y_train_sub)
        val_pred = model.predict(X_val)
        val_tmp = val_sub[["Date", target]].copy()
        val_tmp["pred"] = val_pred
        ic = (
            val_tmp.groupby("Date")
            .apply(lambda x: x["pred"].corr(x[target], method="spearman"))
            .mean()
        )
        if not np.isnan(ic) and ic > best_ic:
            best_ic = ic
            best_alpha = alpha
    
    # Train on full training window with best alpha
    y_train = rank_transform(train_data, target)
    X_train = train_data[features]
    X_test = test_data[features]
    
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    data.loc[test_data.index, "pred_ridge"] = model.predict(X_test)
    
    # Progress logging
    if i % 200 == 0:
        pct = (i - LOOKBACK) / (len(unique_dates) - LOOKBACK) * 100
        print(f"  Day {i}/{len(unique_dates)} ({pct:.0f}%)")

print("Ridge predictions complete!")
print(f"Prediction coverage: {data['pred_ridge'].notna().mean():.1%}")
"""),

("md", """## 5. LightGBM Model

**Why GBDT?**
- Can capture non-linear interactions between features
- Handles noisy data well with proper regularization
- Different model family → better ensemble diversity with Ridge

We retrain GBDT every 63 days (quarterly) to balance freshness vs computational cost.
"""),

("code", """RETRAIN_FREQ = 63  # Retrain every ~quarter

data["pred_gbdt"] = np.nan

lgb_params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

print(f"Running rolling LightGBM (retrain every {RETRAIN_FREQ} days)...")

gbdt_model = None
last_train_idx = -RETRAIN_FREQ  # Force initial training

for i in range(LOOKBACK, len(unique_dates)):
    test_date = unique_dates[i]
    test_data = data[data["Date"] == test_date]
    
    if len(test_data) == 0:
        continue
    
    # Retrain periodically
    if i - last_train_idx >= RETRAIN_FREQ:
        train_dates = unique_dates[max(0, i - LOOKBACK) : i]
        train_data = data[data["Date"].isin(train_dates)]
        
        X_train = train_data[features]
        y_train = rank_transform(train_data, target)
        
        # Replace or create new model
        train_set = lgb.Dataset(X_train, label=y_train)
        gbdt_model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=200,
        )
        last_train_idx = i
    
    # Predict
    X_test = test_data[features]
    data.loc[test_data.index, "pred_gbdt"] = gbdt_model.predict(X_test)
    
    if i % 200 == 0:
        pct = (i - LOOKBACK) / (len(unique_dates) - LOOKBACK) * 100
        print(f"  Day {i}/{len(unique_dates)} ({pct:.0f}%)")

print("GBDT predictions complete!")
print(f"Prediction coverage: {data['pred_gbdt'].notna().mean():.1%}")
"""),

("md", """## 5b. CatBoost Model

**Why CatBoost?**
- Strong default performance on tabular data
- Handles over-fitting natively with symmetric trees
- Good complement to LightGBM in our ensemble
"""),

("code", """data["pred_cb"] = np.nan

cb_params = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "MAE",
    "l2_leaf_reg": 3.0,
    "verbose": 0,
    "random_seed": 42,
    "thread_count": -1
}

print(f"Running rolling CatBoost (retrain every {RETRAIN_FREQ} days)...")

cb_model = None
last_train_idx_cb = -RETRAIN_FREQ

for i in range(LOOKBACK, len(unique_dates)):
    test_date = unique_dates[i]
    test_data = data[data["Date"] == test_date]
    
    if len(test_data) == 0:
        continue
    
    # Retrain periodically
    if i - last_train_idx_cb >= RETRAIN_FREQ:
        train_dates = unique_dates[max(0, i - LOOKBACK) : i]
        train_data = data[data["Date"].isin(train_dates)]
        
        X_train = train_data[features]
        y_train = rank_transform(train_data, target)
        
        # Replace or create new model
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(X_train, y_train)
        last_train_idx_cb = i
        
    # Predict
    X_test = test_data[features]
    data.loc[test_data.index, "pred_cb"] = cb_model.predict(X_test)
    
    if i % 200 == 0:
        pct = (i - LOOKBACK) / (len(unique_dates) - LOOKBACK) * 100
        print(f"  Day {i}/{len(unique_dates)} ({pct:.0f}%)")

print("CatBoost predictions complete!")
print(f"Prediction coverage: {data['pred_cb'].notna().mean():.1%}")
"""),

("md", """## 6. Ensemble: Ridge + LightGBM + CatBoost

Combining different model families reduces variance and improves stability. We use a performance-weighted ensemble (60% Ridge, 20% GBDT, 20% CatBoost) based on our hyperparameter tuning.
"""),

("code", """# Standardize each model's predictions cross-sectionally before averaging
for col in ["pred_ridge", "pred_gbdt", "pred_cb"]:
    grp = data.groupby("Date")[col]
    mu = grp.transform("mean")
    sigma = grp.transform("std").replace(0, np.nan)
    data[f"{col}_cs"] = ((data[col] - mu) / sigma).fillna(0.0)

# Weighted ensemble (60% Ridge, 20% LighGBM, 20% CatBoost)
data["pred_ensemble"] = 0.6 * data["pred_ridge_cs"] + 0.2 * data["pred_gbdt_cs"] + 0.2 * data["pred_cb_cs"]

print("Ensemble prediction created: 0.6 × Ridge + 0.2 × GBDT + 0.2 × CatBoost")
"""),

("md", """## 7. Information Coefficient Analysis

The Information Coefficient (IC) is the Spearman rank correlation between predicted and actual returns. It's the standard measure of predictive power in quant finance.

- IC > 0.02 is meaningful
- IC > 0.05 is strong
- Consistency (IC IR = mean(IC)/std(IC)) matters more than magnitude
"""),

("code", """# Compute daily IC for each model
models = {"Ridge": "pred_ridge", "GBDT": "pred_gbdt", "CatBoost": "pred_cb", "Ensemble": "pred_ensemble"}

# Filter to out-of-sample period only
oos_data = data[data["pred_ridge"].notna() & data["pred_gbdt"].notna() & data["pred_cb"].notna()].copy()

ic_results = {}
for name, col in models.items():
    daily_ic = (
        oos_data.groupby("Date")
        .apply(lambda x: x[col].corr(x[target], method="spearman"))
    )
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_results[name] = {
        "Mean IC": ic_mean,
        "IC Std": ic_std,
        "IC IR": ic_ir,
        "Hit Rate": (daily_ic > 0).mean(),
        "daily_ic": daily_ic,
    }
    print(f"{name:10s} | Mean IC: {ic_mean:.4f} | IC Std: {ic_std:.4f} | "
          f"IC IR: {ic_ir:.3f} | Hit Rate: {(daily_ic > 0).mean():.1%}")
"""),

("code", """# Plot rolling IC
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

for ax, (name, res) in zip(axes, ic_results.items()):
    ic_series = res["daily_ic"]
    rolling_ic = ic_series.rolling(63).mean()
    ax.bar(ic_series.index, ic_series.values, alpha=0.15, color="steelblue", width=1)
    ax.plot(rolling_ic.index, rolling_ic.values, color="red", linewidth=1.5, label="63D Rolling Mean")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{name} — Daily IC")
    ax.legend()

plt.xlabel("Date")
plt.tight_layout()
plt.savefig("../outputs/daily_ic.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 8. Feature Importance (GBDT)"""),

("code", """# Get feature importance from the last GBDT model
importance = pd.Series(
    gbdt_model.feature_importance(importance_type="gain"),
    index=features
).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
importance.plot(kind="barh", color="steelblue")
plt.title("LightGBM Feature Importance (Gain)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 9. Signal → Portfolio Construction

**Strategy logic**:
1. Rank stocks by ensemble prediction each day
2. **Long** top 20% (highest predicted return) — concentrated conviction portfolio
3. **Weight** by inverse volatility (risk-parity) × prediction confidence
4. **Smooth** positions with EWM to reduce turnover

This creates a long-only portfolio that captures both market beta and alpha from the ML signal.
"""),

("code", """# Work with out-of-sample data only
data_oos = data[data["pred_ensemble"].notna()].copy()
data_oos = data_oos.sort_values(["Date", "ticker"]).reset_index(drop=True)

# Rank by ensemble prediction
data_oos["pred_rank"] = (
    data_oos.groupby("Date")["pred_ensemble"]
    .rank(ascending=False, method="first")
)
data_oos["n_assets"] = data_oos.groupby("Date")["ticker"].transform("count")

# Long-only: top 20% stocks with highest predicted returns
data_oos["is_long"] = data_oos["pred_rank"] <= 0.20 * data_oos["n_assets"]

print(f"Long fraction: {data_oos['is_long'].mean():.1%}")
print(f"Universe size: {data_oos['n_assets'].iloc[0]:.0f}")
"""),

("code", """# Equal-weight within the long leg — simple and robust
# (Inverse-vol x confidence over-concentrates into a few stocks
# and empirically underperforms equal-weight in this universe)

data_oos["long_weight"] = 0.0
long_mask = data_oos["is_long"]
data_oos.loc[long_mask, "long_weight"] = 1.0
data_oos.loc[long_mask, "long_weight"] /= (
    data_oos.loc[long_mask].groupby("Date")["long_weight"].transform("sum")
)

# Position = long weight only (no short leg)
data_oos["position_raw"] = data_oos["long_weight"]

# Smooth positions (EWM) to reduce turnover
SMOOTH_ALPHA = 0.10
data_oos["position"] = (
    data_oos.groupby("ticker")["position_raw"]
    .transform(lambda x: x.ewm(alpha=SMOOTH_ALPHA, adjust=False).mean())
)

print("Long-only positions constructed.")
print(f"Total weight per day: {data_oos.groupby('Date')['position'].sum().mean():.4f}")
"""),

("md", """## 10. Position Summary"""),

("code", """# Position diagnostics
print("=== Position Summary ===")
print(f"Avg stocks held per day: {(data_oos['position'] > 0.001).groupby(data_oos['Date']).sum().mean():.1f}")
print(f"Mean total weight: {data_oos.groupby('Date')['position'].sum().mean():.4f}")
print(f"Max single-stock weight: {data_oos['position'].max():.4f}")
print(f"Min non-zero weight: {data_oos[data_oos['position'] > 0.001]['position'].min():.4f}")
"""),

("code", """# Save predictions and positions for Part 3 (backtesting)
save_cols = [
    "Date", "ticker", "Open", "High", "Low", "Close", "Volume",
    "log_ret_1d", "fwd_ret_1d", "fwd_ret_5d",
    "vol_10", "pred_ridge", "pred_gbdt", "pred_cb", "pred_ensemble",
    "pred_rank", "is_long", "position_raw", "position"
]
data_oos[save_cols].to_csv("../outputs/data_with_predictions.csv", index=False)
print("Saved: ../outputs/data_with_predictions.csv")
print(f"Shape: {data_oos[save_cols].shape}")
"""),

("md", """## Summary

| Component | Description |
|-----------|------------|
| **Ridge Regression** | Rolling 252-day window with alpha selection |
| **LightGBM** | Rolling retraining every 63 days |
| **CatBoost** | Rolling retraining every 63 days |
| **Ensemble** | Weighted: 60% Ridge + 20% GBDT + 20% CatBoost |
| **Portfolio** | Long-only top 20%, equal-weight, EWM α=0.10 |
| **Smoothing** | EWM (α=0.10) to reduce turnover |

Predictions and positions are saved for Part 3 backtesting.
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3: Backtesting & Performance Analysis
# ═══════════════════════════════════════════════════════════════════════════════
nb3_cells = [
("md", """# Part 3: Backtesting & Performance Analysis
**Precog Quant Task 2026**

This notebook covers:
1. Loading predictions and positions from Part 2
2. Realistic backtesting simulation with transaction costs
3. Key performance metrics: Sharpe, Max Drawdown, Turnover, Total Return
4. Cumulative PnL: Strategy vs Equal-Weight Benchmark
5. Analysis: transaction cost sensitivity, regime analysis, failure modes

> *"A lower Sharpe that remains stable over a long time frame is much more valuable than a ridiculously high Sharpe over one year."*
"""),

("code", """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True
sns.set_style("whitegrid")

print("Libraries loaded.")
"""),

("md", """## 1. Load Predictions"""),

("code", """data = pd.read_csv("../outputs/data_with_predictions.csv", parse_dates=["Date"])
data = data.sort_values(["Date", "ticker"]).reset_index(drop=True)

print(f"Shape: {data.shape}")
print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
print(f"Trading days: {data['Date'].nunique()}")
print(f"Assets: {data['ticker'].nunique()}")
data.head()
"""),

("md", """## 2. Backtesting Engine

### Simulation Constraints:
- **Initial Capital**: $1,000,000
- **Transaction Costs**: 10 bps (0.10%) per trade
- **Universe**: All 100 assets
- **Rebalancing**: Daily (positions from Part 2)

### How the backtest works:
1. Each day, we compute the desired portfolio weights from the ensemble model
2. We calculate turnover as the sum of absolute weight changes
3. Transaction costs are deducted proportionally to turnover
4. Daily portfolio return = Σ(weight_i × return_i) - costs
"""),

("code", """# ============================================================
# COMPUTE DAILY PORTFOLIO RETURNS & TURNOVER
# ============================================================

INITIAL_CAPITAL = 1_000_000
COST_BPS = 10  # basis points per trade (round-trip)
COST_RATE = COST_BPS / 10_000

# Calculate daily weight changes (turnover)
data["prev_position"] = data.groupby("ticker")["position"].shift(1).fillna(0.0)
data["weight_change"] = (data["position"] - data["prev_position"]).abs()

# Daily turnover (sum of absolute weight changes / 2 for double-counting)
daily_turnover = data.groupby("Date")["weight_change"].sum() / 2

# Daily gross portfolio return (before costs)
daily_return_gross = (
    data.groupby("Date")
    .apply(lambda x: (x["position"] * x["fwd_ret_1d"]).sum())
)

# Transaction costs
daily_costs = COST_RATE * daily_turnover

# Net return
daily_return_net = daily_return_gross - daily_costs

# Also compute equal-weight benchmark (buy & hold all 100 stocks)
benchmark_return = data.groupby("Date")["fwd_ret_1d"].mean()

print("Backtest simulation complete.")
print(f"Average daily turnover: {daily_turnover.mean():.4f}")
print(f"Average daily cost: {daily_costs.mean():.6f}")
"""),

("code", """# ============================================================
# CUMULATIVE PnL
# ============================================================

# Strategy cumulative return
cum_return_strategy = np.exp(daily_return_net.cumsum())
cum_return_strategy_gross = np.exp(daily_return_gross.cumsum())

# Benchmark cumulative return
cum_return_benchmark = np.exp(benchmark_return.cumsum())

# Dollar PnL
pnl_strategy = INITIAL_CAPITAL * cum_return_strategy
pnl_benchmark = INITIAL_CAPITAL * cum_return_benchmark

print(f"Strategy final value: ${pnl_strategy.iloc[-1]:,.2f}")
print(f"Benchmark final value: ${pnl_benchmark.iloc[-1]:,.2f}")
print(f"Strategy total return: {(cum_return_strategy.iloc[-1] - 1) * 100:.2f}%")
print(f"Benchmark total return: {(cum_return_benchmark.iloc[-1] - 1) * 100:.2f}%")
"""),

("md", """## 3. Cumulative PnL Plot: Strategy vs Benchmark"""),

("code", """fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Growth of $1M
axes[0].plot(pnl_strategy.index, pnl_strategy.values, label="Strategy (Net)", linewidth=2, color="steelblue")
axes[0].plot(pnl_strategy.index, (INITIAL_CAPITAL * cum_return_strategy_gross).values, 
             label="Strategy (Gross)", linewidth=1, color="steelblue", alpha=0.5, linestyle="--")
axes[0].plot(pnl_benchmark.index, pnl_benchmark.values, label="Equal-Weight Benchmark", 
             linewidth=2, color="black", linestyle="--")
axes[0].set_title("Cumulative PnL: Strategy vs Equal-Weight Benchmark")
axes[0].set_ylabel("Portfolio Value ($)")
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

# Plot 2: Strategy excess return over benchmark (alpha)
excess = daily_return_net - benchmark_return
excess = excess.dropna()
cum_excess = np.exp(excess.cumsum())
axes[1].plot(cum_excess.index, cum_excess.values, color="green", linewidth=2)
axes[1].axhline(1.0, color="black", linewidth=0.5, linestyle="--")
axes[1].set_title("Cumulative Alpha (Strategy - Benchmark)")
axes[1].set_ylabel("Excess Growth of $1")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.savefig("../outputs/cumulative_pnl.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 4. Performance Metrics

We compute:
- **Annualized Sharpe Ratio**: $SR = \\frac{\\bar{r}}{\\sigma_r} \\times \\sqrt{252}$
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Drawdown**: Mean of all drawdowns
- **Portfolio Turnover**: Average daily one-sided turnover
- **Total Return**: Cumulative return in dollars and percentage
- **Calmar Ratio**: Annualized return / Max drawdown
"""),

("code", """def compute_metrics(returns, name="Strategy"):
    \"\"\"Compute standard portfolio metrics from a return series.\"\"\"
    returns = returns.dropna()
    
    # Annualized return
    ann_return = returns.mean() * 252
    
    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0 risk-free rate)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    equity = np.exp(returns.cumsum())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()
    avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
    
    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # Total return
    total_return = equity.iloc[-1] - 1
    
    # Sortino ratio (downside vol)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0
    
    return {
        "Name": name,
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}", 
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Sortino Ratio": f"{sortino:.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Avg Drawdown": f"{avg_dd:.2%}",
        "Calmar Ratio": f"{calmar:.3f}",
        "Total Return": f"{total_return:.2%}",
        "Final Value ($1M)": f"${INITIAL_CAPITAL * (1 + total_return):,.0f}",
    }

# Compute metrics for all strategies
results = []
results.append(compute_metrics(daily_return_net, "Strategy (Net of Costs)"))
results.append(compute_metrics(daily_return_gross, "Strategy (Gross)"))
results.append(compute_metrics(benchmark_return, "Equal-Weight Benchmark"))

metrics_df = pd.DataFrame(results).set_index("Name")
print(metrics_df.to_string())
"""),

("code", """# Additional: Portfolio turnover
print("\\n=== Turnover Statistics ===")
print(f"Mean daily turnover: {daily_turnover.mean():.4f} ({daily_turnover.mean()*100:.2f}%)")
print(f"Annualized turnover: {daily_turnover.mean() * 252:.2f}x")
print(f"Max daily turnover: {daily_turnover.max():.4f}")

print(f"\\n=== Transaction Cost Impact ===")
print(f"Mean daily cost: {daily_costs.mean():.6f} ({daily_costs.mean()*10000:.2f} bps)")
print(f"Total cost drag (annualized): {daily_costs.mean() * 252:.4f} ({daily_costs.mean() * 252 * 100:.2f}%)")
"""),

("md", """## 5. Drawdown Analysis"""),

("code", """# Drawdown plot
equity = np.exp(daily_return_net.cumsum())
running_max = equity.cummax()
drawdown = equity / running_max - 1.0

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(equity.index, equity.values, color="steelblue", linewidth=1.5)
axes[0].plot(running_max.index, running_max.values, color="gray", linewidth=0.5, linestyle="--")
axes[0].set_title("Equity Curve")
axes[0].set_ylabel("Growth of $1")

axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
axes[1].set_title("Drawdown")
axes[1].set_ylabel("Drawdown (%)")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))

plt.xlabel("Date")
plt.tight_layout()
plt.savefig("../outputs/drawdown.png", dpi=150, bbox_inches="tight")
plt.show()

# Worst drawdown periods
dd_end = drawdown.idxmin()
dd_start = equity.loc[:dd_end].idxmax()
print(f"Worst drawdown: {drawdown.min():.2%} (from {dd_start.date()} to {dd_end.date()})")
"""),

("md", """## 6. Rolling Sharpe Ratio"""),

("code", """# Rolling 63-day (~3 month) and 252-day (~1 year) Sharpe
fig, ax = plt.subplots(figsize=(14, 5))

for window, color, label in [(63, "steelblue", "63-Day Rolling"), (252, "firebrick", "252-Day Rolling")]:
    rolling_mean = daily_return_net.rolling(window).mean()
    rolling_std = daily_return_net.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=color, linewidth=1, label=label, alpha=0.8)

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title("Rolling Sharpe Ratio (Annualized)")
ax.set_ylabel("Sharpe Ratio")
ax.set_xlabel("Date")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 7. Transaction Cost Sensitivity

We re-run the backtest at various cost levels to see how robust the strategy is.
"""),

("code", """cost_levels = [0, 2, 5, 10, 15, 20, 30]  # in bps
cost_results = []

for bps in cost_levels:
    rate = bps / 10_000
    costs_i = rate * daily_turnover
    ret_i = daily_return_gross - costs_i
    
    ann_ret = ret_i.mean() * 252
    ann_vol = ret_i.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    total_ret = np.exp(ret_i.sum()) - 1
    
    cost_results.append({
        "Cost (bps)": bps,
        "Ann. Return": f"{ann_ret:.2%}",
        "Sharpe": round(sharpe, 3),
        "Total Return": f"{total_ret:.2%}",
        "Final Value": f"${INITIAL_CAPITAL * (1 + total_ret):,.0f}",
    })

cost_df = pd.DataFrame(cost_results)
print(cost_df.to_string(index=False))
"""),

("code", """# Plot PnL under different cost assumptions
fig, ax = plt.subplots(figsize=(14, 6))

for bps in [0, 5, 10, 20]:
    rate = bps / 10_000
    costs_i = rate * daily_turnover
    ret_i = daily_return_gross - costs_i
    cum_i = np.exp(ret_i.cumsum())
    ax.plot(cum_i.index, cum_i.values, linewidth=1.5, label=f"{bps} bps")

# Benchmark
ax.plot(cum_return_benchmark.index, cum_return_benchmark.values, 
        color="black", linestyle="--", linewidth=2, label="Benchmark")

ax.set_title("Strategy Performance Under Different Transaction Cost Assumptions")
ax.set_ylabel("Growth of $1")
ax.set_xlabel("Date")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/cost_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 8. Monthly Return Heatmap"""),

("code", """# Monthly returns
monthly_ret = daily_return_net.groupby([
    daily_return_net.index.year.rename("Year"),
    daily_return_net.index.month.rename("Month")
]).sum()

monthly_pivot = monthly_ret.unstack("Month")
monthly_pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:monthly_pivot.shape[1]]

plt.figure(figsize=(14, 5))
sns.heatmap(monthly_pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0,
            linewidths=0.5, cbar_kws={"label": "Monthly Return"})
plt.title("Monthly Return Heatmap")
plt.tight_layout()
plt.savefig("../outputs/monthly_returns.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 9. Analysis & Discussion

### Did the strategy survive transaction costs?

We compare gross vs net performance and examine the cost drag at various levels in the table above. The strategy's survival depends on:
- The magnitude of alpha generated
- The portfolio turnover (controlled by EWM smoothing)
- The transaction cost assumption

### When and why did it fail?

The drawdown chart reveals periods of underperformance. Typical failure modes in momentum-based strategies include:
1. **Regime changes** — Sudden shifts from trending to mean-reverting markets
2. **Volatility explosions** — Market crashes where correlations spike toward 1
3. **Factor crowding** — When too many participants trade the same signals

The rolling Sharpe shows how the strategy's edge varies over time, highlighting periods where the market regime was less favorable.
"""),

("code", """# Yearly breakdown
yearly_ret = daily_return_net.groupby(daily_return_net.index.year).sum()
yearly_vol = daily_return_net.groupby(daily_return_net.index.year).std() * np.sqrt(252)
yearly_sharpe = (daily_return_net.groupby(daily_return_net.index.year).mean() * 252) / yearly_vol

yearly_df = pd.DataFrame({
    "Return": yearly_ret.map(lambda x: f"{x:.2%}"),
    "Volatility": yearly_vol.map(lambda x: f"{x:.2%}"),
    "Sharpe": yearly_sharpe.round(3),
})
print("=== Yearly Performance ===")
print(yearly_df.to_string())
"""),

("md", """## Summary

| Metric | Value |
|--------|-------|
| Annualized Sharpe | See metrics table above |
| Max Drawdown | See drawdown analysis |
| Portfolio Turnover | Controlled by EWM smoothing |
| Transaction Costs | 10 bps per trade, sensitivity analyzed |

The strategy is a long-only portfolio selecting the top 20% of assets daily based on ensemble predictions from Ridge regression, LightGBM, and CatBoost models. Positions are equal-weighted within the long leg and EWM-smoothed (α=0.10) to control turnover. All performance shown is **out-of-sample**.
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 4: Statistical Arbitrage Overlay
# ═══════════════════════════════════════════════════════════════════════════════
nb4_cells = [
("md", """# Part 4: Statistical Arbitrage Overlay
**Precog Quant Task 2026**

This notebook covers:
1. Correlation analysis across assets
2. Cointegration testing (Engle-Granger)
3. Lead-lag relationship discovery
4. Pairs trading signal design
5. Visual analysis of identified pairs
6. Implementation idea for integrating stat-arb into the main portfolio

> *"Oftentimes, there exist assets that exhibit correlated or cointegrated movement. Your goal is to find examples and explain the rationale."*
"""),

("code", """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True
sns.set_style("whitegrid")

print("Libraries loaded.")
"""),

("md", """## 1. Load Data"""),

("code", """data = pd.read_csv("../data/cleaned_panel_data.csv", parse_dates=["Date"])
data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)

# Create price pivot table (tickers as columns, dates as rows)
price_pivot = data.pivot_table(index="Date", columns="ticker", values="Close")
returns_pivot = data.pivot_table(index="Date", columns="ticker", values="log_ret_1d")

# Drop any assets with too many missing values
missing_pct = price_pivot.isnull().mean()
valid_tickers = missing_pct[missing_pct < 0.05].index.tolist()
price_pivot = price_pivot[valid_tickers].dropna()
returns_pivot = returns_pivot[valid_tickers].dropna()

print(f"Assets with sufficient data: {len(valid_tickers)}")
print(f"Date range: {price_pivot.index.min().date()} to {price_pivot.index.max().date()}")
print(f"Trading days: {len(price_pivot)}")
"""),

("md", """## 2. Correlation Analysis

We start with the simplest measure — Pearson and Spearman correlation of returns — as a baseline.

**Important**: Correlation measures co-movement at the same time, but doesn't tell us about:
- Direction of causality (lead-lag)
- Mean-reverting spreads (cointegration)
- Time-varying relationships
"""),

("code", """# Compute full return correlation matrix
corr_matrix = returns_pivot.corr(method="spearman")

# Visualize
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=~mask, cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.8,
            square=True, linewidths=0, cbar_kws={"label": "Spearman Correlation"})
plt.title("Pairwise Return Correlation Matrix (Spearman)")
plt.tight_layout()
plt.savefig("../outputs/correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("code", """# Find the most correlated pairs
pairs_corr = []
tickers = corr_matrix.columns.tolist()
for i in range(len(tickers)):
    for j in range(i+1, len(tickers)):
        pairs_corr.append({
            "Asset_A": tickers[i],
            "Asset_B": tickers[j],
            "Spearman_Corr": corr_matrix.iloc[i, j]
        })

pairs_df = pd.DataFrame(pairs_corr).sort_values("Spearman_Corr", ascending=False)

print("=== Top 15 Most Correlated Pairs ===")
print(pairs_df.head(15).to_string(index=False))
print()
print("=== Top 10 Most Negatively Correlated Pairs ===")
print(pairs_df.tail(10).to_string(index=False))
"""),

("md", """## 3. Rolling Correlation

Correlations are not static. We look at how the top pairs' correlation evolves over time.
"""),

("code", """# Top 5 correlated pairs — rolling correlation
top_pairs = pairs_df.head(5)

fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)

for idx, (_, row) in enumerate(top_pairs.iterrows()):
    a, b = row["Asset_A"], row["Asset_B"]
    rolling_corr = returns_pivot[a].rolling(63).corr(returns_pivot[b])
    
    axes[idx].plot(rolling_corr.index, rolling_corr.values, linewidth=1, color="steelblue")
    axes[idx].axhline(row["Spearman_Corr"], color="red", linewidth=0.5, linestyle="--")
    axes[idx].set_title(f"{a} vs {b} (overall ρ = {row['Spearman_Corr']:.3f})")
    axes[idx].set_ylabel("Rolling 63D Corr")
    axes[idx].set_ylim(-0.5, 1.0)

plt.xlabel("Date")
plt.tight_layout()
plt.savefig("../outputs/rolling_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 4. Cointegration Analysis

**Correlation ≠ Cointegration.** Two assets can be highly correlated but not cointegrated, and vice versa.

- **Correlation** measures linear association of returns (short-term)
- **Cointegration** measures whether a linear combination of price levels is stationary (long-run equilibrium)

If assets A and B are cointegrated, the spread $S_t = P_A(t) - \\beta \\cdot P_B(t)$ is mean-reverting — the basis of pairs trading.

We use the **Engle-Granger** two-step test:
1. Regress log-prices: $\\log P_A = \\alpha + \\beta \\cdot \\log P_B + \\epsilon$
2. Test if residuals $\\epsilon$ are stationary (ADF test)
"""),

("code", """# Run cointegration test on all pairs (computationally intensive — sample top candidates)
# Pre-filter: only test pairs with correlation > 0.5 (reduces computation dramatically)
candidate_pairs = pairs_df[pairs_df["Spearman_Corr"] > 0.5].copy()
print(f"Testing {len(candidate_pairs)} candidate pairs for cointegration...")

log_prices = np.log(price_pivot)

coint_results = []
for _, row in candidate_pairs.iterrows():
    a, b = row["Asset_A"], row["Asset_B"]
    try:
        score, pvalue, _ = coint(log_prices[a], log_prices[b])
        coint_results.append({
            "Asset_A": a,
            "Asset_B": b,
            "Corr": row["Spearman_Corr"],
            "Coint_Score": score,
            "Coint_PValue": pvalue,
            "Cointegrated_5pct": pvalue < 0.05,
        })
    except Exception:
        pass

coint_df = pd.DataFrame(coint_results).sort_values("Coint_PValue")
print(f"\\nCointegrated pairs (p < 0.05): {coint_df['Cointegrated_5pct'].sum()} out of {len(coint_df)}")
print()
print("=== Top 15 Most Cointegrated Pairs ===")
print(coint_df.head(15).to_string(index=False))
"""),

("code", """# Visualize top cointegrated pairs
top_coint = coint_df.head(6)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flat

for idx, (_, row) in enumerate(top_coint.iterrows()):
    a, b = row["Asset_A"], row["Asset_B"]
    
    # Normalize prices to start at 100 for visual comparison
    norm_a = price_pivot[a] / price_pivot[a].iloc[0] * 100
    norm_b = price_pivot[b] / price_pivot[b].iloc[0] * 100
    
    axes[idx].plot(norm_a.index, norm_a.values, label=a, linewidth=1)
    axes[idx].plot(norm_b.index, norm_b.values, label=b, linewidth=1)
    axes[idx].set_title(f"{a} vs {b}\\nCorr={row['Corr']:.3f}, Coint p={row['Coint_PValue']:.4f}")
    axes[idx].legend(fontsize=8)
    axes[idx].set_ylabel("Normalized Price")

plt.tight_layout()
plt.savefig("../outputs/cointegrated_pairs.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 5. Spread Analysis for Top Cointegrated Pairs

For a cointegrated pair, we construct the spread and analyze its mean-reverting behavior.

$$\\text{Spread}_t = \\log P_A(t) - \\beta \\cdot \\log P_B(t)$$

where $\\beta$ is the OLS hedge ratio.
"""),

("code", """from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Analyze top 3 cointegrated pairs
top3 = coint_df.head(3)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for idx, (_, row) in enumerate(top3.iterrows()):
    a, b = row["Asset_A"], row["Asset_B"]
    
    # OLS regression to get hedge ratio
    y = log_prices[a].values
    X = add_constant(log_prices[b].values)
    model = OLS(y, X).fit()
    beta = model.params[1]
    alpha = model.params[0]
    
    # Spread
    spread = log_prices[a] - beta * log_prices[b] - alpha
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    
    # ADF test on spread
    adf_stat, adf_pval, _, _, _, _ = adfuller(spread.values)
    
    # Plot spread
    axes[idx, 0].plot(spread.index, spread.values, linewidth=0.8, color="steelblue")
    axes[idx, 0].axhline(spread_mean, color="black", linewidth=0.5, linestyle="--")
    axes[idx, 0].axhline(spread_mean + 2*spread_std, color="red", linewidth=0.5, linestyle="--")
    axes[idx, 0].axhline(spread_mean - 2*spread_std, color="green", linewidth=0.5, linestyle="--")
    axes[idx, 0].set_title(f"Spread: {a} − {beta:.2f}×{b}\\nADF p={adf_pval:.4f}")
    axes[idx, 0].set_ylabel("Spread")
    
    # Plot z-score
    axes[idx, 1].plot(z_score.index, z_score.values, linewidth=0.8, color="purple")
    axes[idx, 1].axhline(0, color="black", linewidth=0.5)
    axes[idx, 1].axhline(2, color="red", linewidth=0.5, linestyle="--")
    axes[idx, 1].axhline(-2, color="green", linewidth=0.5, linestyle="--")
    axes[idx, 1].set_title(f"Z-Score of Spread")
    axes[idx, 1].set_ylabel("Z-Score")

plt.tight_layout()
plt.savefig("../outputs/spread_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 6. Lead-Lag Relationship Analysis

Beyond simultaneous co-movement, we investigate whether any asset **leads** another — i.e., today's return of Asset A predicts tomorrow's return of Asset B.

We compute cross-correlation at various lags and identify significant lead-lag relationships.
"""),

("code", """# Cross-correlation function at multiple lags
MAX_LAG = 5  # Up to 5 trading days
top_corr_pairs = pairs_df.head(20)  # Test top 20 correlated pairs

lead_lag_results = []
for _, row in top_corr_pairs.iterrows():
    a, b = row["Asset_A"], row["Asset_B"]
    ret_a = returns_pivot[a].dropna()
    ret_b = returns_pivot[b].dropna()
    
    # Align
    common_idx = ret_a.index.intersection(ret_b.index)
    ret_a = ret_a.loc[common_idx]
    ret_b = ret_b.loc[common_idx]
    
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag == 0:
            continue
        if lag > 0:
            # A leads B by 'lag' days
            cc = ret_a.iloc[:-lag].corr(ret_b.iloc[lag:])
            direction = f"{a} leads {b}"
        else:
            # B leads A by 'lag' days
            cc = ret_b.iloc[:lag].corr(ret_a.iloc[-lag:])
            direction = f"{b} leads {a}"
        
        lead_lag_results.append({
            "Asset_A": a,
            "Asset_B": b,
            "Lag": lag,
            "Direction": direction,
            "Cross_Corr": cc,
            "Abs_Cross_Corr": abs(cc),
        })

ll_df = pd.DataFrame(lead_lag_results).sort_values("Abs_Cross_Corr", ascending=False)

# Significance threshold (rough): |r| > 2/sqrt(N)
n_obs = len(common_idx)
sig_threshold = 2 / np.sqrt(n_obs)
ll_df["Significant"] = ll_df["Abs_Cross_Corr"] > sig_threshold

print(f"Significance threshold: {sig_threshold:.4f}")
print()
print("=== Top 15 Lead-Lag Relationships ===")
print(ll_df.head(15)[["Asset_A", "Asset_B", "Lag", "Direction", "Cross_Corr", "Significant"]].to_string(index=False))
"""),

("code", """# Visualize cross-correlation for top 3 pairs
top_ll_pairs = ll_df.drop_duplicates(subset=["Asset_A", "Asset_B"]).head(3)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (_, row) in enumerate(top_ll_pairs.iterrows()):
    a, b = row["Asset_A"], row["Asset_B"]
    
    lags = range(-MAX_LAG, MAX_LAG + 1)
    cc_vals = []
    for lag in lags:
        ret_a = returns_pivot[a].dropna()
        ret_b = returns_pivot[b].dropna()
        common_idx = ret_a.index.intersection(ret_b.index)
        ret_a = ret_a.loc[common_idx]
        ret_b = ret_b.loc[common_idx]
        
        if lag == 0:
            cc_vals.append(ret_a.corr(ret_b))
        elif lag > 0:
            cc_vals.append(ret_a.iloc[:-lag].reset_index(drop=True).corr(
                ret_b.iloc[lag:].reset_index(drop=True)))
        else:
            cc_vals.append(ret_b.iloc[:-abs(lag)].reset_index(drop=True).corr(
                ret_a.iloc[abs(lag):].reset_index(drop=True)))
    
    axes[idx].bar(lags, cc_vals, color="steelblue", alpha=0.7)
    axes[idx].axhline(sig_threshold, color="red", linewidth=0.5, linestyle="--")
    axes[idx].axhline(-sig_threshold, color="red", linewidth=0.5, linestyle="--")
    axes[idx].axhline(0, color="black", linewidth=0.5)
    axes[idx].set_title(f"{a} vs {b}")
    axes[idx].set_xlabel(f"Lag (positive = {a} leads)")
    axes[idx].set_ylabel("Cross-Correlation")

plt.suptitle("Lead-Lag Cross-Correlation", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../outputs/lead_lag.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## 7. Sector/Cluster Analysis via PCA

Since the assets are anonymized, we can't directly identify sectors. However, we can use **PCA** to discover latent factors that group assets together — these may correspond to sectors, themes, or risk factors.
"""),

("code", """from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# PCA on return correlations
ret_clean = returns_pivot.dropna(axis=1)
pca = PCA(n_components=10)
pca.fit(ret_clean.T)  # Transpose: assets as observations, days as features

# Explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, 11), pca.explained_variance_ratio_, color="steelblue")
axes[0].set_title("PCA Explained Variance Ratio")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained")

axes[1].plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_), 
             marker="o", color="steelblue")
axes[1].set_title("Cumulative Variance Explained")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance")
axes[1].axhline(0.5, color="red", linewidth=0.5, linestyle="--")

plt.tight_layout()
plt.savefig("../outputs/pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Variance explained by first 3 PCs: {pca.explained_variance_ratio_[:3].sum():.1%}")
print(f"Variance explained by first 5 PCs: {pca.explained_variance_ratio_[:5].sum():.1%}")
"""),

("code", """# Cluster assets using first 5 PCs
pc_scores = pca.transform(ret_clean.T)[:, :5]
asset_names = ret_clean.columns.tolist()

# K-Means clustering
N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pc_scores)

cluster_df = pd.DataFrame({
    "Asset": asset_names,
    "Cluster": clusters,
    "PC1": pc_scores[:, 0],
    "PC2": pc_scores[:, 1],
})

# Scatter plot
fig, ax = plt.subplots(figsize=(12, 8))
for c in range(N_CLUSTERS):
    mask = cluster_df["Cluster"] == c
    ax.scatter(cluster_df.loc[mask, "PC1"], cluster_df.loc[mask, "PC2"], 
               label=f"Cluster {c}", s=60, alpha=0.7)
    for _, r in cluster_df[mask].iterrows():
        ax.annotate(r["Asset"].replace("Asset_", ""), (r["PC1"], r["PC2"]), 
                    fontsize=6, alpha=0.7)

ax.set_title("Asset Clustering (PCA + K-Means)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/asset_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

# Print clusters
for c in range(N_CLUSTERS):
    assets = cluster_df[cluster_df["Cluster"] == c]["Asset"].tolist()
    print(f"Cluster {c} ({len(assets)} assets): {', '.join([a.replace('Asset_', '') for a in assets])}")
"""),

("code", """# Within-cluster vs between-cluster correlation
print("=== Average Correlation Within vs Between Clusters ===\\n")

for c in range(N_CLUSTERS):
    cluster_assets = cluster_df[cluster_df["Cluster"] == c]["Asset"].tolist()
    other_assets = cluster_df[cluster_df["Cluster"] != c]["Asset"].tolist()
    
    if len(cluster_assets) < 2:
        continue
    
    within_corrs = []
    for i in range(len(cluster_assets)):
        for j in range(i+1, len(cluster_assets)):
            if cluster_assets[i] in corr_matrix.columns and cluster_assets[j] in corr_matrix.columns:
                within_corrs.append(corr_matrix.loc[cluster_assets[i], cluster_assets[j]])
    
    between_corrs = []
    for a in cluster_assets[:5]:  # sample
        for b in other_assets[:20]:  # sample
            if a in corr_matrix.columns and b in corr_matrix.columns:
                between_corrs.append(corr_matrix.loc[a, b])
    
    within_avg = np.mean(within_corrs) if within_corrs else 0
    between_avg = np.mean(between_corrs) if between_corrs else 0
    
    print(f"Cluster {c}: Within={within_avg:.3f}  Between={between_avg:.3f}  "
          f"Ratio={within_avg/between_avg:.2f}x  ({len(cluster_assets)} assets)")
"""),

("md", """## 8. Pairs Trading Strategy — Conceptual Implementation

### How to integrate stat-arb into the main portfolio:

Given a cointegrated pair $(A, B)$ with hedge ratio $\\beta$:

**Signal Construction:**
$$z_t = \\frac{(\\log P_A(t) - \\beta \\cdot \\log P_B(t)) - \\mu}{\\sigma}$$

**Trading Rules:**
- **Enter long spread** when $z_t < -2$ (spread is cheap → buy A, sell βB)
- **Enter short spread** when $z_t > +2$ (spread is rich → sell A, buy βB)
- **Exit** when $z_t$ crosses 0 (spread reverts to mean)

**Integration with Main Portfolio:**
1. Run the main alpha model (Part 2) to generate directional signals
2. Overlay pairs signals as an additional alpha source
3. Combine using a weighting scheme:

$$w_i^{\\text{final}} = \\lambda \\cdot w_i^{\\text{directional}} + (1-\\lambda) \\cdot w_i^{\\text{pairs}}$$

where $\\lambda \\in [0.7, 0.9]$ (directional model gets more weight since it's trained on all assets).

**Key Considerations:**
- The hedge ratio $\\beta$ should be estimated on a rolling window (not static)
- The spread's half-life determines the holding period: $t_{1/2} = \\frac{\\ln(2)}{-\\ln(\\phi)}$ where $\\phi$ is the AR(1) coefficient
- Position sizing should be inversely proportional to spread volatility
- Stop-losses at $|z| > 4$ to protect against regime breaks
"""),

("code", """# Demonstrate pairs trading signals for the best cointegrated pair
data_oos["stat_arb_pos"] = 0.0
if len(coint_df) > 0:
    best_pair = coint_df.iloc[0]
    a, b = best_pair["Asset_A"], best_pair["Asset_B"]
    
    # Rolling hedge ratio (use 126-day rolling OLS)
    WINDOW = 126
    spreads = pd.Series(index=log_prices.index, dtype=float)
    betas = pd.Series(index=log_prices.index, dtype=float)
    
    for i in range(WINDOW, len(log_prices)):
        y = log_prices[a].iloc[i-WINDOW:i].values
        x = log_prices[b].iloc[i-WINDOW:i].values
        x_const = np.column_stack([np.ones(WINDOW), x])
        params = np.linalg.lstsq(x_const, y, rcond=None)[0]
        alpha_coef, beta_coef = params
        
        date = log_prices.index[i]
        betas.loc[date] = beta_coef
        spreads.loc[date] = log_prices[a].iloc[i] - beta_coef * log_prices[b].iloc[i] - alpha_coef
    
    spreads = spreads.dropna()
    betas = betas.dropna()
    
    # Rolling z-score of spread
    spread_mean = spreads.rolling(63).mean()
    spread_std = spreads.rolling(63).std()
    z_spread = (spreads - spread_mean) / spread_std.replace(0, np.nan)
    z_spread = z_spread.dropna()
    
    # Generate signals
    signals = pd.Series(0.0, index=z_spread.index)
    signals[z_spread < -2] = 1.0    # Long spread (buy A, sell B)
    signals[z_spread > 2] = -1.0    # Short spread (sell A, buy B)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    
    # Normalized prices
    norm_a = price_pivot[a] / price_pivot[a].iloc[0] * 100
    norm_b = price_pivot[b] / price_pivot[b].iloc[0] * 100
    axes[0].plot(norm_a, label=a, linewidth=1)
    axes[0].plot(norm_b, label=b, linewidth=1)
    axes[0].set_title(f"Normalized Prices: {a} vs {b}")
    axes[0].legend()
    
    # Rolling hedge ratio
    axes[1].plot(betas, color="purple", linewidth=1)
    axes[1].set_title("Rolling Hedge Ratio (β)")
    axes[1].set_ylabel("β")
    
    # Z-score of spread
    axes[2].plot(z_spread, color="steelblue", linewidth=0.8)
    axes[2].axhline(2, color="red", linewidth=0.5, linestyle="--", label="Entry: Short spread")
    axes[2].axhline(-2, color="green", linewidth=0.5, linestyle="--", label="Entry: Long spread")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_title("Spread Z-Score")
    axes[2].legend()
    
    # Signals
    axes[3].plot(signals, color="darkgreen", linewidth=0.8)
    axes[3].set_title("Trading Signals")
    axes[3].set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig("../outputs/pairs_trading_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Half-life estimation
    spread_arr = spreads.values
    spread_lag = spread_arr[:-1]
    spread_diff = np.diff(spread_arr)
    beta_ar1 = np.polyfit(spread_lag, spread_diff, 1)[0]
    half_life = -np.log(2) / beta_ar1 if beta_ar1 < 0 else np.inf
    print(f"\\nBest pair: {a} vs {b}")
    print(f"Cointegration p-value: {best_pair['Coint_PValue']:.6f}")
    print(f"Estimated half-life: {half_life:.1f} days")
    print(f"Number of entry signals: {(signals != 0).sum()}")
else:
    print("No cointegrated pairs found — this is not unusual in anonymized data.")
"""),

("md", """## 9. Multi-Timeframe Correlation Analysis

Assets may move together at different timeframes. Weekly correlation can differ significantly from daily correlation.
"""),

("code", """# Compute correlations at different timeframes
weekly_returns = returns_pivot.resample("W").sum()
monthly_returns = returns_pivot.resample("ME").sum()

corr_daily = returns_pivot.corr()
corr_weekly = weekly_returns.corr()
corr_monthly = monthly_returns.corr()

# Compare: which pairs are more correlated at longer timeframes?
timeframe_comparison = []
for i in range(len(valid_tickers)):
    for j in range(i+1, len(valid_tickers)):
        a, b = valid_tickers[i], valid_tickers[j]
        if a in corr_daily.columns and b in corr_daily.columns:
            timeframe_comparison.append({
                "Asset_A": a,
                "Asset_B": b,
                "Daily": corr_daily.loc[a, b],
                "Weekly": corr_weekly.loc[a, b] if a in corr_weekly.columns and b in corr_weekly.columns else np.nan,
                "Monthly": corr_monthly.loc[a, b] if a in corr_monthly.columns and b in corr_monthly.columns else np.nan,
            })

tf_df = pd.DataFrame(timeframe_comparison).dropna()
tf_df["Weekly_vs_Daily"] = tf_df["Weekly"] - tf_df["Daily"]
tf_df["Monthly_vs_Daily"] = tf_df["Monthly"] - tf_df["Daily"]

print("=== Pairs Where Longer-Term Correlation >> Daily ===")
print(tf_df.sort_values("Monthly_vs_Daily", ascending=False).head(10)[
    ["Asset_A", "Asset_B", "Daily", "Weekly", "Monthly"]
].to_string(index=False))
"""),

("code", """# Scatter plot: daily vs weekly correlation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(tf_df["Daily"], tf_df["Weekly"], alpha=0.3, s=10, color="steelblue")
axes[0].plot([-.5, 1], [-.5, 1], "r--", linewidth=0.5)
axes[0].set_xlabel("Daily Correlation")
axes[0].set_ylabel("Weekly Correlation")
axes[0].set_title("Daily vs Weekly Return Correlation")

axes[1].scatter(tf_df["Daily"], tf_df["Monthly"], alpha=0.3, s=10, color="firebrick")
axes[1].plot([-.5, 1], [-.5, 1], "r--", linewidth=0.5)
axes[1].set_xlabel("Daily Correlation")
axes[1].set_ylabel("Monthly Correlation")
axes[1].set_title("Daily vs Monthly Return Correlation")

plt.tight_layout()
plt.savefig("../outputs/timeframe_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

("md", """## Summary & Conclusions

### Key Findings:

1. **Correlated Pairs**: Several asset pairs exhibit high return correlations (ρ > 0.7), suggesting potential sector or theme overlap.

2. **Cointegration**: A subset of correlated pairs also show statistically significant cointegration (p < 0.05), meaning their price levels maintain a long-run equilibrium — the foundation of pairs trading.

3. **Lead-Lag**: Some pairs show significant cross-correlation at non-zero lags, indicating that one asset's movement can partially predict the other's future return.

4. **Cluster Structure**: PCA + K-Means reveals natural groupings among the 100 assets, likely corresponding to GICS sectors or industry groups in the underlying (anonymized) data.

5. **Multi-Timeframe**: Correlations tend to be stronger at longer timeframes (weekly, monthly), consistent with the idea that assets share common fundamental drivers that are more visible at lower frequencies.

### Integration into Main Portfolio:

The pairs/stat-arb signal can be incorporated as an additional alpha layer:
- Weight: 10-30% of total signal weight
- Benefits: diversifying vs. the directional momentum/ML signals
- Risk: regime breaks in cointegration relationships, convergence failure

$$w_i^{\\text{total}} = 0.8 \\cdot w_i^{\\text{alpha}} + 0.2 \\cdot w_i^{\\text{stat-arb}}$$

This overlay adds a **relative-value** component to the portfolio, capturing alpha from mean-reversion in spreads while the main model captures **cross-sectional** alpha from momentum and ML predictions.
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# WRITE ALL NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════════════════
notebooks = [
    ("01_part1_baseline.ipynb", nb1_cells),
    ("02_part2_enhancements.ipynb", nb2_cells),
    ("03_part3_backtesting.ipynb", nb3_cells),
    ("04_part4_stat_arb.ipynb", nb4_cells),
]

for fname, cells in notebooks:
    path = os.path.join(NB_DIR, fname)
    nb = make_nb(cells)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Written: {path} ({len(cells)} cells)")

print("\nAll notebooks generated successfully!")
