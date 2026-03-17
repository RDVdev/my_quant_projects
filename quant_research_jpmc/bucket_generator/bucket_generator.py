import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

NUM_BUCKETS = 5


def load_fico_data(filepath):
    df = pd.read_csv(filepath)
    fico_scores = df["fico_score"].values
    defaults = df["default"].values

    print(f"Loaded {len(df)} borrowers")
    print(f"FICO range: {fico_scores.min()}-{fico_scores.max()}")
    print(f"Default rate: {defaults.mean():.2%}")
    return df, fico_scores, defaults


def print_default_rates_by_range(fico_scores, defaults):
    ranges = [(300, 600), (600, 650), (650, 700), (700, 750), (750, 850)]
    print("\nDefault rates by FICO range:")
    for low, high in ranges:
        mask = (fico_scores >= low) & (fico_scores < high)
        if mask.sum() > 0:
            print(f"  {low}-{high}: {defaults[mask].mean():.2%} ({mask.sum()} borrowers)")


def calculate_mse(boundaries, fico_scores):
    boundaries = sorted([fico_scores.min()] + list(boundaries) + [fico_scores.max()])
    total_mse = 0

    for i in range(len(boundaries) - 1):
        mask = (fico_scores >= boundaries[i]) & (fico_scores <= boundaries[i + 1])
        bucket_scores = fico_scores[mask]
        if len(bucket_scores) > 0:
            total_mse += ((bucket_scores - bucket_scores.mean()) ** 2).sum()

    return total_mse


def calculate_log_likelihood(boundaries, fico_scores, defaults):
    boundaries = sorted([fico_scores.min()] + list(boundaries) + [fico_scores.max()])
    ll = 0

    for i in range(len(boundaries) - 1):
        mask = (fico_scores >= boundaries[i]) & (fico_scores <= boundaries[i + 1])
        n_i = mask.sum()
        k_i = defaults[mask].sum()

        if n_i > 0 and k_i > 0 and k_i < n_i:
            ll += n_i * np.log(n_i) - k_i * np.log(k_i) - (n_i - k_i) * np.log(n_i - k_i)

    return ll


def optimize_mse_buckets(fico_scores, num_buckets):
    min_s, max_s = fico_scores.min(), fico_scores.max()
    initial = np.linspace(min_s, max_s, num_buckets + 1)[1:-1]
    bounds = [(min_s + 1, max_s - 1)] * (num_buckets - 1)

    result = minimize(lambda b: calculate_mse(b, fico_scores), initial, method="L-BFGS-B", bounds=bounds)
    return sorted(result.x), result.fun


def optimize_ll_buckets(fico_scores, defaults, num_buckets):
    min_s, max_s = fico_scores.min(), fico_scores.max()
    initial = np.linspace(min_s, max_s, num_buckets + 1)[1:-1]
    bounds = [(min_s + 1, max_s - 1)] * (num_buckets - 1)

    result = minimize(lambda b: -calculate_log_likelihood(b, fico_scores, defaults), initial, method="L-BFGS-B", bounds=bounds)
    return sorted(result.x), -result.fun


def print_buckets(boundaries, fico_scores, defaults, method_name):
    boundaries_full = [fico_scores.min()] + list(boundaries) + [fico_scores.max()]
    print(f"\n=== {method_name} Buckets ===")
    print(f"Boundaries: {[int(b) for b in boundaries]}")
    for i in range(len(boundaries_full) - 1):
        low, high = int(boundaries_full[i]), int(boundaries_full[i + 1])
        mask = (fico_scores >= low) & (fico_scores <= high)
        dr = defaults[mask].mean() if mask.sum() > 0 else 0
        print(f"  Bucket {i + 1}: {low}-{high} | Default: {dr:.2%} | Count: {mask.sum()}")


def fico_to_bucket(score, boundaries):
    for i, b in enumerate(sorted(boundaries)):
        if score < b:
            return i + 1
    return len(boundaries) + 1


def plot_comparison(fico_scores, mse_boundaries, ll_boundaries, save_path="bucket_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, (name, bounds) in zip(axes, [("MSE", mse_boundaries), ("Log-Likelihood", ll_boundaries)]):
        ax.hist(fico_scores, bins=50, alpha=0.5, edgecolor="black")
        for b in bounds:
            ax.axvline(b, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("FICO Score")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} Bucketing")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nSaved: {save_path}")


def run_bucket_generator():
    df, fico_scores, defaults = load_fico_data("Task 3 and 4_Loan_Data.csv")
    print_default_rates_by_range(fico_scores, defaults)

    # MSE optimization
    mse_boundaries, mse_val = optimize_mse_buckets(fico_scores, NUM_BUCKETS)
    print_buckets(mse_boundaries, fico_scores, defaults, "MSE")
    print(f"MSE: {mse_val:,.2f}")

    # Log-Likelihood optimization
    ll_boundaries, ll_val = optimize_ll_buckets(fico_scores, defaults, NUM_BUCKETS)
    print_buckets(ll_boundaries, fico_scores, defaults, "Log-Likelihood")
    print(f"Log-Likelihood: {ll_val:.2f}")

    # comparison
    print("\n=== Method Comparison ===")
    comparison = pd.DataFrame({
        "Method": ["MSE", "Log-Likelihood"],
        "Boundaries": [[int(b) for b in mse_boundaries], [int(b) for b in ll_boundaries]],
    })
    print(comparison.to_string(index=False))

    plot_comparison(fico_scores, mse_boundaries, ll_boundaries)

    # test the mapping
    final_boundaries = ll_boundaries  # LL is better for credit risk
    test_scores = [550, 650, 700, 750, 820]
    print("\n=== FICO → Bucket Mapping ===")
    for s in test_scores:
        print(f"  FICO {s} → Bucket {fico_to_bucket(s, final_boundaries)}")


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
            run_bucket_generator()
        finally:
            _sys.stdout = _orig
