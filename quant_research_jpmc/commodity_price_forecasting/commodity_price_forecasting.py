import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

DAYS_PER_YEAR = 365.25


def load_gas_data(filepath):
    df = pd.read_csv(filepath)
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)

    df["Month"] = df["Dates"].dt.month
    df["Year"] = df["Dates"].dt.year
    df["Day"] = df["Dates"].dt.dayofyear

    # cyclical features — map day-of-year to sine/cosine wave
    df["sin_day"] = np.sin(2 * np.pi * df["Day"] / DAYS_PER_YEAR)
    df["cos_day"] = np.cos(2 * np.pi * df["Day"] / DAYS_PER_YEAR)

    print(f"Loaded {len(df)} monthly price records ({df['Dates'].min().date()} to {df['Dates'].max().date()})")
    return df


def train_price_model(df):
    X = df[["Year", "sin_day", "cos_day"]].values
    y = df["Prices"].values

    model = LinearRegression()
    model.fit(X, y)
    print(f"Model R²: {model.score(X, y):.4f}")
    return model


def estimate_price(date, model):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    day_of_year = date.timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * day_of_year / DAYS_PER_YEAR)
    cos_day = np.cos(2 * np.pi * day_of_year / DAYS_PER_YEAR)

    X_pred = np.array([[date.year, sin_day, cos_day]])
    return model.predict(X_pred)[0]


def plot_historical(df, save_path="historical_prices.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Dates"], df["Prices"], "o-", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Natural Gas Price")
    plt.title("Monthly Natural Gas Prices")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


def plot_seasonal_pattern(df, save_path="seasonal_pattern.png"):
    monthly_avg = df.groupby("Month")["Prices"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_avg.index, monthly_avg.values)
    plt.xlabel("Month")
    plt.ylabel("Average Price")
    plt.title("Seasonal Pattern: Average Price by Month")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


def plot_extrapolation(df, model, save_path="extrapolation.png"):
    last_date = df["Dates"].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=30),
        end=last_date + timedelta(days=365),
        freq="ME",
    )
    future_prices = [estimate_price(d, model) for d in future_dates]

    plt.figure(figsize=(14, 7))
    plt.plot(df["Dates"], df["Prices"], "o-", label="Historical", linewidth=2)
    plt.plot(future_dates, future_prices, "s--", label="Extrapolated", linewidth=2, color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Natural Gas Prices: Historical + 1-Year Extrapolation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


def run_commodity_forecasting():
    df = load_gas_data("Nat_Gas.csv")
    model = train_price_model(df)

    plot_historical(df)
    plot_seasonal_pattern(df)
    plot_extrapolation(df, model)

    # demo predictions
    test_dates = ["2025-01-15", "2025-06-15", "2025-12-15"]
    print("\n=== Price Estimates ===")
    for d in test_dates:
        price = estimate_price(d, model)
        print(f"  {d}: ${price:.2f}")


if __name__ == "__main__":
    import sys as _sys

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    _orig = _sys.stdout
    with open("output.txt", "w") as _f:
        _sys.stdout = _Tee(_orig, _f)
        try:
            run_commodity_forecasting()
        finally:
            _sys.stdout = _orig
