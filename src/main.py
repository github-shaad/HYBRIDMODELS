from model_factory import ModelFactory as ensem
from strategies import MLStrategy, HybridStrategy, ArimaStrategy
from preprocessors import LogDiffByRow, RowWiseMinMaxScaler, NormalizeByRow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from config.config import *
from plots import MultiPlot, BacktestPlot
from storageManager import StorageManager
from backtesting import *


if __name__ == "__main__":
    train_data_path = RAW_DATA_DIR / "train_data_raw.npy"
    test_data_path = RAW_DATA_DIR / "test_data_raw.npy"
    a = np.load(train_data_path)
    b = np.load(test_data_path)

    respath = PREDICTIONS_DIR / "model_predictions"

    res = np.load(f"{respath}/arima.npy")
    data_dict = {}
    for i in range(25):
        data_dict[str(i)] = (b[i], res[i])

    num_assets = 25
    train_days = 2011
    test_days = 503
    total_days = train_days + test_days 

    dummy_market_caps = np.random.uniform(low=100, high=1000, size=(total_days, num_assets))

    backtestingCurve = BackTesting(dummy_market_caps, a, b, res, 2.5, 100, 30).equityCurve(100)
    print(backtestingCurve)
    backtestingDict = {"Arima": backtestingCurve}
    s = StorageManager()
    arimaPlots = MultiPlot("Arima")
    arimaPlots.plot(data_dict, cols=5)
    backtest = BacktestPlot("Arima - Backtesting Equity Curve", "blue")
    backtest.plot(backtestingDict, (10,5))
    
    s.store_predictions(backtestingCurve, "portfolio", "arima_raw_data")
    s.store_figures(arimaPlots, "model", "arima_raw_data")
    s.store_figures(backtest, "portfolio", "arima_raw_data")
