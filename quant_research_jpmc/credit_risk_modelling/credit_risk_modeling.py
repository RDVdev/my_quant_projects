import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report

RECOVERY_RATE = 0.10
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_explore(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Defaulters: {df['default'].sum()}, Non-defaulters: {(df['default']==0).sum()}")
    return df


def plot_distributions(df, save_path="data_exploration.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    cols = [
        ("income", "Income Distribution by Default Status"),
        ("fico_score", "FICO Score Distribution by Default Status"),
        ("total_debt_outstanding", "Debt Distribution by Default Status"),
        ("years_employed", "Employment Duration by Default Status"),
    ]

    for ax, (col, title) in zip(axes.flat, cols):
        ax.hist(
            [df[df["default"] == 0][col], df[df["default"] == 1][col]],
            label=["No Default", "Default"],
            bins=30,
        )
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


def prepare_data(df):
    X = df.drop(["customer_id", "default"], axis=1)
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(name, y_test, y_pred, y_pred_proba):
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | Log Loss: {ll:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    return {"Model": name, "Accuracy": acc, "ROC-AUC": auc, "Log Loss": ll}


def train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols):
    # Logistic Regression (needs scaled features)
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1_000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred, lr_proba)

    # Random Forest (raw features)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_proba)

    # Feature importance
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("Feature Importance (RF):")
    print(importance.to_string(index=False))

    # Comparison
    comparison = pd.DataFrame([lr_metrics, rf_metrics])
    best = comparison.loc[comparison["ROC-AUC"].idxmax(), "Model"]
    print(f"\nBest model (ROC-AUC): {best}")

    plot_model_comparison(comparison)
    return lr, rf, comparison


def plot_model_comparison(comparison, save_path="model_comparison.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, metric in enumerate(["Accuracy", "ROC-AUC", "Log Loss"]):
        axes[i].bar(comparison["Model"], comparison[metric], color=["blue", "green"])
        axes[i].set_ylabel(metric)
        axes[i].set_title(f"{metric} Comparison")
        axes[i].set_ylim([0, max(comparison[metric]) * 1.2])
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


def predict_expected_loss(borrower_data, loan_amount, model, scaler, feature_cols, recovery_rate=RECOVERY_RATE):
    """
    Predict expected loss for a loan.
    Expected Loss = PD × EAD × LGD
    """
    input_df = pd.DataFrame([borrower_data])[feature_cols]
    input_scaled = scaler.transform(input_df)
    pd_prob = model.predict_proba(input_scaled)[0][1]
    lgd = 1 - recovery_rate
    expected_loss = pd_prob * loan_amount * lgd
    return expected_loss, pd_prob


def run_predictions(model, scaler, feature_cols):
    test_cases = {
        "Risky Borrower": {
            "data": {
                "credit_lines_outstanding": 8,
                "loan_amt_outstanding": 45_000,
                "total_debt_outstanding": 80_000,
                "income": 30_000,
                "years_employed": 1,
                "fico_score": 520,
            },
            "loan": 15_000,
        },
        "Good Borrower": {
            "data": {
                "credit_lines_outstanding": 2,
                "loan_amt_outstanding": 10_000,
                "total_debt_outstanding": 10_000,
                "income": 130_000,
                "years_employed": 2,
                "fico_score": 800,
            },
            "loan": 15_000,
        },
        "Average Borrower": {
            "data": {
                "credit_lines_outstanding": 3,
                "loan_amt_outstanding": 20_000,
                "total_debt_outstanding": 30_000,
                "income": 80_000,
                "years_employed": 7,
                "fico_score": 750,
            },
            "loan": 15_000,
        },
    }

    print("\n=== Predictions ===")
    for label, case in test_cases.items():
        loss, pd_prob = predict_expected_loss(
            case["data"], case["loan"], model, scaler, feature_cols
        )
        print(f"{label}: PD={pd_prob:.2%}, Expected Loss=${loss:,.2f}")


def run_credit_risk_model():
    df = load_and_explore("Task 3 and 4_Loan_Data.csv")
    plot_distributions(df)
    X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)
    lr_model, rf_model, comparison = train_models(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    )
    run_predictions(lr_model, scaler, X.columns)


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
            run_credit_risk_model()
        finally:
            _sys.stdout = _orig