from config.config import * 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import os
from metrics import Metrics
MODEL_NAME = "VARMAX-LSTM"
N_RUNS = 5000
PERCENT_SIG = 0.1
model_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{MODEL_NAME}.npy")
sandp = np.load(RAW_DATA_DIR / "sandp_prices.npy")[-len(model_equity_curve):]
def drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
    running_max = np.where(running_max == 0, 1e-8, running_max)
        
    drawdown_curve = (equity_curve - running_max) / running_max

    return drawdown_curve

def sandp_equity_curve(index, capital):
    sandp_equity_curve = (index / index[0]) * capital
    return sandp_equity_curve
def get_sharpe(returns):
    # Protect against flatlines (std = 0)
    if np.std(returns) == 0:
        return 0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

sandp_equity_curve = sandp_equity_curve(sandp, 10_000)

model_returns = np.diff(model_equity_curve) / model_equity_curve[:-1]
sandp_returns = np.diff(sandp_equity_curve) / sandp_equity_curve[:-1]

actual_diff_sharpe = Metrics.sharpe(model_equity_curve) - Metrics.sharpe(sandp_equity_curve)
actual_diff_sortino = Metrics.sortino(model_equity_curve) - Metrics.sortino(sandp_equity_curve)
actual_diff_cagr = Metrics.cagr(model_equity_curve) - Metrics.cagr(sandp_equity_curve)
actual_diff_mdd = Metrics.mdd(model_equity_curve) - Metrics.mdd(sandp_equity_curve)
actual_diff_calmar = Metrics.calmar(model_equity_curve) - Metrics.calmar(sandp_equity_curve)

random_diffs_sharpe = []
random_diffs_sortino = []
random_diffs_cagr = []
random_diffs_mdd = []
random_diffs_calmar = []


print(f"\nRunning {N_RUNS} permutations...")
for i in range(N_RUNS):
    print(MODEL_NAME)
    print(f"Iteration : {i+1}")
    swap_mask = np.random.randint(0, 2, size=len(model_returns)).astype(bool)
    
    # Create fake return streams by swapping days between your model and the S&P 500
    fake_strat = np.where(swap_mask, sandp_returns, model_returns)
    fake_sp500 = np.where(swap_mask, model_returns, sandp_returns)
    fake_strat_equity = np.insert(np.cumprod(1 + fake_strat), 0, 1.0)
    fake_sp500_equity = np.insert(np.cumprod(1 + fake_sp500), 0, 1.0)
    
    # Calculate fake difference
    random_diffs_sharpe.append(Metrics.sharpe(fake_strat_equity) - Metrics.sharpe(fake_sp500_equity))
    random_diffs_sortino.append(Metrics.sortino(fake_strat_equity) - Metrics.sortino(fake_sp500_equity))
    random_diffs_cagr.append(Metrics.cagr(fake_strat_equity) - Metrics.cagr(fake_sp500_equity))
    random_diffs_mdd.append(Metrics.mdd(fake_strat_equity) - Metrics.mdd(fake_sp500_equity))
    random_diffs_calmar.append(Metrics.calmar(fake_strat_equity) - Metrics.calmar(fake_sp500_equity))
    os.system('cls' if os.name == 'nt' else 'clear')
p_value_sharpe = np.sum(np.array(random_diffs_sharpe) >= actual_diff_sharpe) / N_RUNS
p_value_sortino = np.sum(np.array(random_diffs_sortino) >= actual_diff_sortino) / N_RUNS
p_value_cagr = np.sum(np.array(random_diffs_cagr) >= actual_diff_cagr) / N_RUNS
p_value_mdd = np.sum(np.array(random_diffs_mdd) >= actual_diff_mdd) / N_RUNS
p_value_calmar = np.sum(np.array(random_diffs_calmar) >= actual_diff_calmar) / N_RUNS

p_vals = [("sharpe",p_value_sharpe), ("sortino",p_value_sortino),
          ("cagr", p_value_cagr), ("mdd",p_value_mdd), ("calmar", p_value_calmar)]


for p in p_vals:
    print(f"P_val {p[0]} = {p[1]}")
    if p[1] < PERCENT_SIG:
        print(f"Reject Null. {p[0]} Statistically different from S and P")
    else:
        print(f"{p[0]} fails to be statistically different from SandP")