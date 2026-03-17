import pandas as pd
import numpy as np

def evaluate_baseline():
    print("Loading data_with_predictions.csv...")
    try:
        data = pd.read_csv("outputs/data_with_predictions.csv", parse_dates=["Date"])
    except:
        print("Could not find outputs/data_with_predictions.csv")
        return
        
    data = data.sort_values(["Date", "ticker"]).reset_index(drop=True)
    
    INITIAL_CAPITAL = 1_000_000
    COST_RATE = 10 / 10_000
    
    # Strategy returns
    data["prev_position"] = data.groupby("ticker")["position"].shift(1).fillna(0.0)
    data["weight_change"] = (data["position"] - data["prev_position"]).abs()
    
    daily_turnover = data.groupby("Date")["weight_change"].sum() / 2
    daily_return_gross = data.groupby("Date").apply(lambda x: (x["position"] * x["fwd_ret_1d"]).sum())
    daily_costs = COST_RATE * daily_turnover
    daily_return_net = daily_return_gross - daily_costs
    
    cum_return_strategy = np.exp(daily_return_net.cumsum())
    pnl_strategy = INITIAL_CAPITAL * cum_return_strategy
    total_return = (cum_return_strategy.iloc[-1] - 1) * 100
    
    ann_return = daily_return_net.mean() * 252
    ann_vol = daily_return_net.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Equal-weight benchmark
    bench_ret = data.groupby("Date")["fwd_ret_1d"].mean()
    cum_bench = np.exp(bench_ret.cumsum())
    bench_total = (cum_bench.iloc[-1] - 1) * 100
    bench_ann = bench_ret.mean() * 252
    bench_vol = bench_ret.std() * np.sqrt(252)
    bench_sharpe = bench_ann / bench_vol if bench_vol > 0 else 0
    
    print("=== Strategy Metrics ===")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Final Value: ${pnl_strategy.iloc[-1]:,.2f}")
    print(f"Annualized Return: {ann_return*100:.2f}%")
    print(f"Annualized Volatility: {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Mean Daily Turnover: {daily_turnover.mean()*100:.2f}%")
    
    print("\n=== Equal-Weight Benchmark ===")
    print(f"Total Return: {bench_total:.2f}%")
    print(f"Final Value: ${INITIAL_CAPITAL * cum_bench.iloc[-1]:,.2f}")
    print(f"Annualized Return: {bench_ann*100:.2f}%")
    print(f"Sharpe Ratio: {bench_sharpe:.3f}")
    
    print(f"\n=== Strategy vs Benchmark ===")
    alpha = total_return - bench_total
    print(f"Alpha: {alpha:+.2f}%")
    print(f"Beats Benchmark: {'YES' if total_return > bench_total else 'NO'}")

if __name__ == "__main__":
    evaluate_baseline()
