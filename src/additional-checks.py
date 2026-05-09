from config.config import * 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from metrics import Metrics
from scipy.stats import skew
MODEL_NAME = "VARMAX-MLP"
print(MODEL_NAME)
equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{MODEL_NAME}.npy")[4:]
sandp_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / "S_and_P_500.npy")[-len(equity_curve):]

def returns(curve):
    returns = np.diff(curve) / curve[:-1]
    return returns
equity_returns = np.diff(equity_curve) / equity_curve[:-1]
print(f"Skew of returns : {skew(equity_returns)}")

#First 126 days test to rule out the effect of global offset
capital = equity_curve[0]
returns_exclude_days = equity_returns[126:]
exclude_curve = capital*np.cumprod(1 + returns_exclude_days)
#print(f"Sortino excluding 126 days {Metrics.sortino(exclude_curve)}")


#truncation test, Truncating the top 1% returns 
upper_bound = np.percentile(equity_returns, 99)
truncated_returns = np.where(equity_returns < upper_bound, equity_returns, 0)
capital = equity_curve[0]
truncated_equity_curve = capital * np.cumprod(1 + truncated_returns)
truncated_equity_curve = np.insert(truncated_equity_curve, 0, capital)

print(f"Truncated CAGR:{Metrics.cagr(truncated_equity_curve)}")

rolling_metric = []
positive_sortino = 0
for i in range(len(equity_curve) - 60):
    metric = Metrics.sortino(truncated_equity_curve[i:60+i])
    if metric > 0:
        positive_sortino +=1
    rolling_metric.append(metric)
print(f"Positive Sortino {positive_sortino / (len(equity_curve) - 60)}")
#plt.figure(figsize=(10,6))
#plt.plot(rolling_metric)
#plt.show()

#Return Correlation check
print(f"COrr - {np.corrcoef(returns(sandp_equity_curve), returns(equity_curve))}")