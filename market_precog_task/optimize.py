import numpy as np
import pandas as pd
from itertools import product

def run_optimization():
    print("Loading data...")
    # Read the data with out-of-sample predictions
    data = pd.read_csv("outputs/data_with_predictions.csv", parse_dates=["Date"])
    data = data.sort_values(["Date", "ticker"]).reset_index(drop=True)
    
    smooth_alphas = [0.05, 0.1, 0.2]
    quantiles = [0.10, 0.15, 0.20]
    ridge_weights = [0.4, 0.5, 0.6]
    signal_multipliers = [1, -1] # test flipping the signal!
    
    results = []
    
    INITIAL_CAPITAL = 1_000_000
    COST_RATE = 10 / 10_000
    
    for col in ["pred_ridge", "pred_gbdt", "pred_cb"]:
        grp = data.groupby("Date")[col]
        mu = grp.transform("mean")
        sigma = grp.transform("std").replace(0, np.nan)
        data[f"{col}_cs"] = ((data[col] - mu) / sigma).fillna(0.0)
        
    vol_floor = 1e-8
    data["inv_vol"] = 1.0 / data["vol_10"].clip(lower=vol_floor)
    
    dates = data["Date"].unique()
    
    for rw, sa, q, sm in product(ridge_weights, smooth_alphas, quantiles, signal_multipliers):
        # 1. Ensemble prediction
        gw = (1.0 - rw) / 2.0
        cw = (1.0 - rw) / 2.0
        data["pred_ensemble"] = (rw * data["pred_ridge_cs"] + gw * data["pred_gbdt_cs"] + cw * data["pred_cb_cs"]) * sm
        
        # 2. Ranking
        data["pred_rank"] = data.groupby("Date")["pred_ensemble"].rank(ascending=False, method="first")
        data["n_assets"] = data.groupby("Date")["ticker"].transform("count")
        
        # 3. Cutoffs
        data["is_long"] = data["pred_rank"] <= q * data["n_assets"]
        data["is_short"] = data["pred_rank"] > (1 - q) * data["n_assets"]
        
        data["confidence"] = data["pred_ensemble"].abs()
        
        # 4. Weighting
        data["long_weight"] = 0.0
        data.loc[data["is_long"], "long_weight"] = data.loc[data["is_long"], "inv_vol"] * data.loc[data["is_long"], "confidence"]
        data["long_weight"] /= data.groupby("Date")["long_weight"].transform("sum").replace(0, 1)
        
        data["short_weight"] = 0.0
        data.loc[data["is_short"], "short_weight"] = data.loc[data["is_short"], "inv_vol"] * data.loc[data["is_short"], "confidence"]
        data["short_weight"] /= data.groupby("Date")["short_weight"].transform("sum").replace(0, 1)
        data["short_weight"] *= -1.0
        
        data["position_raw"] = data["long_weight"] + data["short_weight"]
        
        # 5. Smoothing
        data["position"] = data.groupby("ticker")["position_raw"].transform(lambda x: x.ewm(alpha=sa, adjust=False).mean())
        
        # 6. Returns and Costs
        data["prev_position"] = data.groupby("ticker")["position"].shift(1).fillna(0.0)
        data["weight_change"] = (data["position"] - data["prev_position"]).abs()
        
        daily_turnover = data.groupby("Date")["weight_change"].sum() / 2
        daily_return_gross = data.groupby("Date").apply(lambda x: (x["position"] * x["fwd_ret_1d"]).sum())
        daily_costs = COST_RATE * daily_turnover
        daily_return_net = daily_return_gross - daily_costs
        
        # 7. Metrics
        ann_return = daily_return_net.mean() * 252
        ann_vol = daily_return_net.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        total_ret = np.exp(daily_return_net.sum()) - 1
        
        results.append({
            "Signal_Mult": sm,
            "Ridge_Wt": rw,
            "Smooth_Alpha": sa,
            "Quantile": q,
            "Ann_Return": ann_return,
            "Sharpe": sharpe,
            "Turnover": daily_turnover.mean(),
            "Total_Ret": total_ret
        })
        
    res_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    print("\nTop 10 Combinations by Sharpe:")
    print(res_df.head(10).to_string(index=False))

if __name__ == "__main__":
    run_optimization()
