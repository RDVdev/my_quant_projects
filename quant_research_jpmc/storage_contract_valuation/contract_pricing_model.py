import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

DAYS_PER_YEAR = 365.25

# default contract parameters
DEFAULT_INJECTION_RATE = 0.50    # $/MMBtu
DEFAULT_WITHDRAWAL_RATE = 0.50   # $/MMBtu
DEFAULT_STORAGE_RATE = 10_000    # $/month
DEFAULT_MAX_STORAGE = 500_000    # MMBtu


def load_and_train_price_model(filepath):
    df = pd.read_csv(filepath)
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)

    df["Day"] = df["Dates"].dt.dayofyear
    df["Year"] = df["Dates"].dt.year
    df["sin_day"] = np.sin(2 * np.pi * df["Day"] / DAYS_PER_YEAR)
    df["cos_day"] = np.cos(2 * np.pi * df["Day"] / DAYS_PER_YEAR)

    X = df[["Year", "sin_day", "cos_day"]].values
    y = df["Prices"].values

    model = LinearRegression().fit(X, y)
    print(f"Price model trained (R²={model.score(X, y):.4f})")
    return model


def estimate_price(date, model):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    day_of_year = date.timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * day_of_year / DAYS_PER_YEAR)
    cos_day = np.cos(2 * np.pi * day_of_year / DAYS_PER_YEAR)

    X_pred = np.array([[date.year, sin_day, cos_day]])
    return model.predict(X_pred)[0]


def price_contract(injection_dates, withdrawal_dates,
                   injection_volumes, withdrawal_volumes,
                   model,
                   injection_rate=DEFAULT_INJECTION_RATE,
                   withdrawal_rate=DEFAULT_WITHDRAWAL_RATE,
                   storage_rate=DEFAULT_STORAGE_RATE,
                   max_storage=DEFAULT_MAX_STORAGE):
    """
    Calculate the value of a gas storage contract.
    Contract Value = Sale Revenue - Purchase Cost - Storage Cost
    """
    first_injection = min(injection_dates)
    last_withdrawal = max(withdrawal_dates)

    # purchase costs (market price + injection fee)
    purchase_cost = sum(
        estimate_price(d, model) * v + injection_rate * v
        for d, v in zip(injection_dates, injection_volumes)
    )

    # sale revenue (market price - withdrawal fee)
    sale_revenue = sum(
        estimate_price(d, model) * v - withdrawal_rate * v
        for d, v in zip(withdrawal_dates, withdrawal_volumes)
    )

    # storage duration
    storage_months = (
        (last_withdrawal.year - first_injection.year) * 12
        + (last_withdrawal.month - first_injection.month)
    )
    storage_cost = storage_rate * storage_months

    # validate inventory constraints
    transactions = []
    for d, v in zip(injection_dates, injection_volumes):
        transactions.append((d, v, "injection"))
    for d, v in zip(withdrawal_dates, withdrawal_volumes):
        transactions.append((d, -v, "withdrawal"))
    transactions.sort(key=lambda x: x[0])

    current_inventory = 0
    for d, v, t_type in transactions:
        current_inventory += v
        if current_inventory < 0:
            raise ValueError(f"Inventory goes negative on {d.date()} (tried to withdraw more than available)")
        if current_inventory > max_storage:
            raise ValueError(f"Inventory exceeds capacity on {d.date()} ({current_inventory:.0f} > {max_storage:.0f})")

    if current_inventory != 0:
        print(f"Warning: {current_inventory:.0f} MMBtu remains in storage at contract end")

    contract_value = sale_revenue - purchase_cost - storage_cost
    return contract_value


def run_contract_valuation():
    model = load_and_train_price_model("Nat_Gas.csv")

    # example: buy in summer (cheap), sell in winter (expensive)
    injection_dates = [pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-01")]
    injection_volumes = [100_000, 150_000, 100_000]

    withdrawal_dates = [pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")]
    withdrawal_volumes = [100_000, 150_000, 100_000]

    print("\n=== Contract Parameters ===")
    print(f"Injection:  {len(injection_dates)} dates, {sum(injection_volumes):,.0f} MMBtu total")
    print(f"Withdrawal: {len(withdrawal_dates)} dates, {sum(withdrawal_volumes):,.0f} MMBtu total")
    print(f"Injection rate: ${DEFAULT_INJECTION_RATE}/MMBtu")
    print(f"Withdrawal rate: ${DEFAULT_WITHDRAWAL_RATE}/MMBtu")
    print(f"Storage rate: ${DEFAULT_STORAGE_RATE:,.0f}/month")
    print(f"Max capacity: {DEFAULT_MAX_STORAGE:,.0f} MMBtu")

    # print price estimates at each date
    print("\n=== Price Estimates ===")
    for d in injection_dates:
        print(f"  Buy  {d.date()}: ${estimate_price(d, model):.2f}/MMBtu")
    for d in withdrawal_dates:
        print(f"  Sell {d.date()}: ${estimate_price(d, model):.2f}/MMBtu")

    try:
        value = price_contract(
            injection_dates, withdrawal_dates,
            injection_volumes, withdrawal_volumes,
            model,
        )
        print(f"\nContract Value: ${value:,.2f}")
    except ValueError as e:
        print(f"\nContract Error: {e}")


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
            run_contract_valuation()
        finally:
            _sys.stdout = _orig
