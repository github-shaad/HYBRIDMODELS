from config.config import * 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from metrics import Metrics

MODEL_NAME = "VARMAX-MLP"
print(MODEL_NAME)

var_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / "VARMAX.npy")
model_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{MODEL_NAME}.npy")[-len(var_equity_curve):]
model_returns = np.diff(model_equity_curve) / model_equity_curve[:-1]
var_returns = np.diff(var_equity_curve) / var_equity_curve[:-1]

actual_diff_sharpe = Metrics.sharpe(model_equity_curve) - Metrics.sharpe(var_equity_curve)
actual_diff_sortino = Metrics.sortino(model_equity_curve) - Metrics.sortino(var_equity_curve)
actual_diff_cagr = Metrics.cagr(model_equity_curve) - Metrics.cagr(var_equity_curve)
actual_diff_mdd = Metrics.mdd(model_equity_curve) - Metrics.mdd(var_equity_curve)
actual_diff_calmar = Metrics.calmar(model_equity_curve) - Metrics.calmar(var_equity_curve)

random_diffs_sharpe = []
random_diffs_sortino = []
random_diffs_cagr = []
random_diffs_mdd = []
random_diffs_calmar = []

N_RUNS = 5000
print(f"\nRunning {N_RUNS} permutations...")
for i in range(N_RUNS):
    # Create a random boolean mask (50/50 chance to swap each day's return)
    swap_mask = np.random.randint(0, 2, size=len(model_returns)).astype(bool)
    
    # Create fake return streams by swapping days between your model and the S&P 500
    fake_strat = np.where(swap_mask, var_returns, model_returns)
    fake_sp500 = np.where(swap_mask, model_returns, var_returns)
    fake_strat_equity = np.insert(np.cumprod(1 + fake_strat), 0, 1.0)
    fake_sp500_equity = np.insert(np.cumprod(1 + fake_sp500), 0, 1.0)
    
    # Calculate fake difference
    random_diffs_sharpe.append(Metrics.sharpe(fake_strat_equity) - Metrics.sharpe(fake_sp500_equity))
    random_diffs_sortino.append(Metrics.sortino(fake_strat_equity) - Metrics.sortino(fake_sp500_equity))
    random_diffs_cagr.append(Metrics.cagr(fake_strat_equity) - Metrics.cagr(fake_sp500_equity))
    random_diffs_mdd.append(Metrics.mdd(fake_strat_equity) - Metrics.mdd(fake_sp500_equity))
    random_diffs_calmar.append(Metrics.calmar(fake_strat_equity) - Metrics.calmar(fake_sp500_equity))

p_value_sharpe = np.sum(np.array(random_diffs_sharpe) >= actual_diff_sharpe) / N_RUNS
p_value_sortino = np.sum(np.array(random_diffs_sortino) >= actual_diff_sortino) / N_RUNS
p_value_cagr = np.sum(np.array(random_diffs_cagr) >= actual_diff_cagr) / N_RUNS
p_value_mdd = np.sum(np.array(random_diffs_mdd) >= actual_diff_mdd) / N_RUNS
p_value_calmar = np.sum(np.array(random_diffs_calmar) >= actual_diff_calmar) / N_RUNS

p_vals = [("sharpe",p_value_sharpe), ("sortino",p_value_sortino),
          ("cagr", p_value_cagr), ("mdd",p_value_mdd), ("calmar", p_value_calmar)]


for p in p_vals:
    print(f"P_val {p[0]} = {p[1]}")
    if p[1] < 0.05:
        print(f"Reject Null. {p[0]} Statistically different from VARMAX")
    else:
        print(f"{p[0]} fails to be statistically different from VARMAX")