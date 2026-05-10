from config.config import * 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
from metrics import Metrics
from backtesting import RankedBackTesting2
import os
"""
MONTE CARLO STATISTICAL SIGNIFICANCE TEST OF SIGNALS. WE TEST IF OUR MODEL SIGNALS ARE STATISTICALLY DIFFERENT FROM 
RANDOM SIGNALS.
"""
MODEL_NAME = "VARMAX-LSTM"
PERCENT_SIG = 0.05
N_RUNS = 5000
offset_for_not_lstm = 4
if MODEL_NAME == "VARMAX-LSTM":
    offset_for_not_lstm = 0
model_predictions = np.load(PREDICTIONS_DIR / "model_predictions" / f"{MODEL_NAME}.npy")[:,offset_for_not_lstm:] 
model_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{MODEL_NAME}.npy")[offset_for_not_lstm:]
prices = np.load(RAW_DATA_DIR / "PRICES.npy")[:,:]


def generate_signals(true_prices, predicted_prices):
    """
    Converts predicted prices into an expected return signal matrix.
    Both inputs should be shape (N_STOCKS, Time)
    """
    today_price = true_prices[:, :-1]
    tomorrow_predicted = predicted_prices[:, 1:]
    expected_returns = (tomorrow_predicted - today_price) / today_price
    return expected_returns, today_price

def shuffle_cross_sectional(signal_matrix):
    """
    Shuffles the signals among stocks for each individual day.
    This destroys stock-picking edge while keeping daily market volatility intact.
    """
    return np.apply_along_axis(np.random.permutation, 0, signal_matrix)
def smooth_exact_shape(data_matrix, window=5, method='ema'):
    df = pd.DataFrame(data_matrix.T)
    if method == 'ema':
        smoothed_df = df.ewm(span=window, adjust=False).mean()
    elif method == 'sma':
        smoothed_df = df.rolling(window=window, min_periods=1).mean()
    smoothed_df = smoothed_df.bfill().ffill()
    result_matrix = smoothed_df.values.T
    assert result_matrix.shape == data_matrix.shape, f"Shape mismatch! Expected {data_matrix.shape}, got {result_matrix.shape}"
    return result_matrix
signals, tradable_prices = generate_signals(prices, model_predictions)
strict_signals = smooth_exact_shape(signals, 5, 'ema')

actual_sharpe = Metrics.sharpe(model_equity_curve)
actual_sortino = Metrics.sortino(model_equity_curve)
actual_cagr = Metrics.cagr(model_equity_curve)
actual_mdd = Metrics.mdd(model_equity_curve)
actual_calmar = Metrics.calmar(model_equity_curve)

sharpes = []
sortinos = []
cagrs = []
mdds = []
calmars = []


for i in range(N_RUNS):
    print(MODEL_NAME)
    print(f"Iteration:{i+1}")
    random_signals = shuffle_cross_sectional(strict_signals)

    rbacktest = RankedBackTesting2(random_signals, tradable_prices, 3,3, 0.001)
    requity_curve, drawdown_curve = rbacktest.equityDrawdownCurve(10_000)

    sharpes.append(Metrics.sharpe(requity_curve))
    sortinos.append(Metrics.sortino(requity_curve))
    cagrs.append(Metrics.cagr(requity_curve))
    mdds.append(Metrics.cagr(requity_curve))
    calmars.append(Metrics.calmar(requity_curve))
    os.system('cls' if os.name == 'nt' else 'clear')

p_value_sharpe = np.sum(np.array(sharpes) >= actual_sharpe) / N_RUNS
p_value_sortino = np.sum(np.array(sortinos) >= actual_sortino) / N_RUNS
p_value_cagr = np.sum(np.array(cagrs) >= actual_cagr) / N_RUNS
p_value_mdd = np.sum(np.array(mdds) >= actual_mdd) / N_RUNS
p_value_calmar = np.sum(np.array(calmars) >= actual_calmar) / N_RUNS

p_vals = [("sharpe",p_value_sharpe), ("sortino",p_value_sortino),
          ("cagr", p_value_cagr), ("mdd",p_value_mdd), ("calmar", p_value_calmar)]


for p in p_vals:
    print(f"P_val {p[0]} = {p[1]}")
    if p[1] < PERCENT_SIG:
        print(f"Reject Null. {p[0]} Statistically different from Random Signals")
    else:
        print(f"{p[0]} fails to be statistically different Random Signals")