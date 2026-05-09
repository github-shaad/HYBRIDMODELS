from model_factory import ModelFactory as models
from config.config import * 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from plots import BacktestEquityDrawdownPlot
from metrics import Metrics
from storageManager import StorageManager


MODEL_NAME = "S_and_P_500"
CURVE = "VARMAX-LSTM"
lstm_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{CURVE}.npy")
sandp = np.load(RAW_DATA_DIR / "sandp_prices.npy")[-len(lstm_equity_curve):]


def drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
    running_max = np.where(running_max == 0, 1e-8, running_max)
        
    drawdown_curve = (equity_curve - running_max) / running_max

    return drawdown_curve

def sandp_equity_curve(index, capital):
    sandp_equity_curve = (index / index[0]) * capital
    return sandp_equity_curve

sandp_equity_curve = sandp_equity_curve(sandp, 10_000)
sandp_drawdown_curve = drawdown(sandp_equity_curve)
lstm_drawdown_curve = drawdown(lstm_equity_curve)



Metrics.save_portfolio_metrics(MODEL_NAME, sandp_equity_curve)
btPlot = BacktestEquityDrawdownPlot(f"S AND P 500 vs {CURVE}")
btPlot.plot({MODEL_NAME:(sandp_equity_curve, sandp_drawdown_curve),
             CURVE:(lstm_equity_curve, lstm_drawdown_curve)})
StorageManager.store_figures(btPlot, "portfolio", f"{MODEL_NAME}vs{CURVE}")
StorageManager.store_predictions(sandp_equity_curve, "portfolio", MODEL_NAME)

CURVE = "VARMAX"
lstm_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{CURVE}.npy")
sandp = np.load(RAW_DATA_DIR / "sandp_prices.npy")[-len(lstm_equity_curve):]


def drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
    running_max = np.where(running_max == 0, 1e-8, running_max)
        
    drawdown_curve = (equity_curve - running_max) / running_max

    return drawdown_curve

def sandp_equity_curve(index, capital):
    sandp_equity_curve = (index / index[0]) * capital
    return sandp_equity_curve

sandp_equity_curve = sandp_equity_curve(sandp, 10_000)
sandp_drawdown_curve = drawdown(sandp_equity_curve)
lstm_drawdown_curve = drawdown(lstm_equity_curve)



Metrics.save_portfolio_metrics(MODEL_NAME, sandp_equity_curve)
btPlot = BacktestEquityDrawdownPlot(f"S AND P 500 vs {CURVE}")
btPlot.plot({MODEL_NAME:(sandp_equity_curve, sandp_drawdown_curve),
             CURVE:(lstm_equity_curve, lstm_drawdown_curve)})
StorageManager.store_figures(btPlot, "portfolio", f"{MODEL_NAME}vs{CURVE}")
StorageManager.store_predictions(sandp_equity_curve, "portfolio", MODEL_NAME)

CURVE = "VARMAX-GBR"
lstm_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{CURVE}.npy")
sandp = np.load(RAW_DATA_DIR / "sandp_prices.npy")[-len(lstm_equity_curve):]


def drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
    running_max = np.where(running_max == 0, 1e-8, running_max)
        
    drawdown_curve = (equity_curve - running_max) / running_max

    return drawdown_curve

def sandp_equity_curve(index, capital):
    sandp_equity_curve = (index / index[0]) * capital
    return sandp_equity_curve

sandp_equity_curve = sandp_equity_curve(sandp, 10_000)
sandp_drawdown_curve = drawdown(sandp_equity_curve)
lstm_drawdown_curve = drawdown(lstm_equity_curve)



Metrics.save_portfolio_metrics(MODEL_NAME, sandp_equity_curve)
btPlot = BacktestEquityDrawdownPlot(f"S AND P 500 vs {CURVE}")
btPlot.plot({MODEL_NAME:(sandp_equity_curve, sandp_drawdown_curve),
             CURVE:(lstm_equity_curve, lstm_drawdown_curve)})
StorageManager.store_figures(btPlot, "portfolio", f"{MODEL_NAME}vs{CURVE}")
StorageManager.store_predictions(sandp_equity_curve, "portfolio", MODEL_NAME)

CURVE = "VARMAX-LGBM"
lstm_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{CURVE}.npy")
sandp = np.load(RAW_DATA_DIR / "sandp_prices.npy")[-len(lstm_equity_curve):]


def drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
    running_max = np.where(running_max == 0, 1e-8, running_max)
        
    drawdown_curve = (equity_curve - running_max) / running_max

    return drawdown_curve

def sandp_equity_curve(index, capital):
    sandp_equity_curve = (index / index[0]) * capital
    return sandp_equity_curve

sandp_equity_curve = sandp_equity_curve(sandp, 10_000)
sandp_drawdown_curve = drawdown(sandp_equity_curve)
lstm_drawdown_curve = drawdown(lstm_equity_curve)



Metrics.save_portfolio_metrics(MODEL_NAME, sandp_equity_curve)
btPlot = BacktestEquityDrawdownPlot(f"S AND P 500 vs {CURVE}")
btPlot.plot({MODEL_NAME:(sandp_equity_curve, sandp_drawdown_curve),
             CURVE:(lstm_equity_curve, lstm_drawdown_curve)})
StorageManager.store_figures(btPlot, "portfolio", f"{MODEL_NAME}vs{CURVE}")
StorageManager.store_predictions(sandp_equity_curve, "portfolio", MODEL_NAME)

CURVE = "VARMAX-SVR"
lstm_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{CURVE}.npy")
sandp = np.load(RAW_DATA_DIR / "sandp_prices.npy")[-len(lstm_equity_curve):]


def drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
    running_max = np.where(running_max == 0, 1e-8, running_max)
        
    drawdown_curve = (equity_curve - running_max) / running_max

    return drawdown_curve

def sandp_equity_curve(index, capital):
    sandp_equity_curve = (index / index[0]) * capital
    return sandp_equity_curve

sandp_equity_curve = sandp_equity_curve(sandp, 10_000)
sandp_drawdown_curve = drawdown(sandp_equity_curve)
lstm_drawdown_curve = drawdown(lstm_equity_curve)



Metrics.save_portfolio_metrics(MODEL_NAME, sandp_equity_curve)
btPlot = BacktestEquityDrawdownPlot(f"S AND P 500 vs {CURVE}")
btPlot.plot({MODEL_NAME:(sandp_equity_curve, sandp_drawdown_curve),
             CURVE:(lstm_equity_curve, lstm_drawdown_curve)})
StorageManager.store_figures(btPlot, "portfolio", f"{MODEL_NAME}vs{CURVE}")
StorageManager.store_predictions(sandp_equity_curve, "portfolio", MODEL_NAME)

CURVE = "VARMAX-MLP"
lstm_equity_curve = np.load(PREDICTIONS_DIR / "portfolio_predictions" / f"{CURVE}.npy")
sandp = np.load(RAW_DATA_DIR / "sandp_prices.npy")[-len(lstm_equity_curve):]


def drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
    running_max = np.where(running_max == 0, 1e-8, running_max)
        
    drawdown_curve = (equity_curve - running_max) / running_max

    return drawdown_curve

def sandp_equity_curve(index, capital):
    sandp_equity_curve = (index / index[0]) * capital
    return sandp_equity_curve

sandp_equity_curve = sandp_equity_curve(sandp, 10_000)
sandp_drawdown_curve = drawdown(sandp_equity_curve)
lstm_drawdown_curve = drawdown(lstm_equity_curve)



Metrics.save_portfolio_metrics(MODEL_NAME, sandp_equity_curve)
btPlot = BacktestEquityDrawdownPlot(f"S AND P 500 vs {CURVE}")
btPlot.plot({MODEL_NAME:(sandp_equity_curve, sandp_drawdown_curve),
             CURVE:(lstm_equity_curve, lstm_drawdown_curve)})
StorageManager.store_figures(btPlot, "portfolio", f"{MODEL_NAME}vs{CURVE}")
StorageManager.store_predictions(sandp_equity_curve, "portfolio", MODEL_NAME)