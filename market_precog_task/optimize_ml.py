import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import Ridge
import lightgbm as lgb
from scipy import stats

def run_ml_optimization():
    print("Loading data...")
    # Because we need to retrain models, we need the cleaned_panel_data.csv
    data = pd.read_csv("data/cleaned_panel_data.csv", parse_dates=["Date"])
    data = data.sort_values(["ticker", "Date"]).reset_index(drop=True)
    
    features = [
        "mom_5_cs", "mom_10_cs", "mom_20_cs", "mom_60_cs",
        "vol_5_cs", "vol_10_cs", "vol_20_cs", "vol_60_cs", "vol_ratio_5_20_cs",
        "rel_volume_cs", "vol_mom_5_cs", "log_dollar_vol_cs",
        "reversal_1d_cs", "dist_ma_20_cs", "dist_ma_60_cs",
        "rsi_14_cs", "bb_width_cs", "bb_pctB_cs", "atr_pct_cs", "hl_range_cs",
        "overnight_ret_cs", "intraday_ret_cs"
    ]
    target = "fwd_ret_5d"
    
    data = data.dropna(subset=[target]).reset_index(drop=True)
    
    unique_dates = np.sort(data["Date"].unique())
    n_dates = len(unique_dates)
    TEST_DAYS = 504
    train_end_idx = n_dates - TEST_DAYS
    train_end_date = unique_dates[train_end_idx]
    
    test_data = data[data["Date"] > train_end_date].copy()
    train_data = data[data["Date"] <= train_end_date].copy()
    
    print(f"Test data shape: {test_data.shape}, Train data shape: {train_data.shape}")
    
    # We will test a simplified walk-forward or just a single train/test split for hyperparam speed
    # To quickly test LightGBM hyperparams, let's just train on the last 500 days of the training set
    # and evaluate on the first 100 days of the test set.
    val_train_dates = unique_dates[train_end_idx - 500 : train_end_idx]
    val_test_dates = unique_dates[train_end_idx + 1 : train_end_idx + 101]
    
    val_train = data[data["Date"].isin(val_train_dates)]
    val_test = data[data["Date"].isin(val_test_dates)]
    
    def rank_transform(df, col):
        return df.groupby("Date")[col].rank(pct=True)
        
    X_train = val_train[features]
    y_train = rank_transform(val_train, target)
    
    X_test = val_test[features]
    y_test = val_test[target]
    
    lgb_params_grid = [
        {"learning_rate": 0.01, "num_leaves": 15, "max_depth": 4},
        {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 5}, # baseline
        {"learning_rate": 0.05, "num_leaves": 63, "max_depth": 6},
        {"learning_rate": 0.01, "num_leaves": 31, "max_depth": 6},
    ]
    
    results = []
    
    for p in lgb_params_grid:
        params = {
            "objective": "regression",
            "metric": "mae",
            "min_child_samples": 50,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            **p
        }
        
        train_set = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_set, num_boost_round=300)
        
        preds = model.predict(X_test)
        
        val_tmp = val_test[["Date", target]].copy()
        val_tmp["pred"] = preds
        
        # Calculate Information Coefficient
        daily_ic = val_tmp.groupby("Date").apply(lambda x: x["pred"].corr(x[target], method="spearman"))
        mean_ic = daily_ic.mean()
        ic_ir = mean_ic / daily_ic.std() if daily_ic.std() > 0 else 0
        
        print(f"Params: {p} -> Mean IC: {mean_ic:.5f}, IC IR: {ic_ir:.3f}")
        results.append({"params": p, "mean_ic": mean_ic, "ic_ir": ic_ir})
        
    res_df = pd.DataFrame(results).sort_values("mean_ic", ascending=False)
    print("\nBest Parameter Sets:")
    print(res_df)

if __name__ == "__main__":
    run_ml_optimization()
