from model_factory import ModelFactory as models
from strategies import MLStrategy, HybridStrategy, ArimaStrategy
from preprocessors import LogDiffByRow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from config.config import *
from plots import MultiPlot, BacktestEquityPlot
from storageManager import StorageManager
from backtesting import *
from metrics import Metrics

if __name__ == "__main__":
    
    #Example Hybrid Strategy
    train_data_path = RAW_DATA_DIR / "train_data_raw.npy"
    test_data_path = RAW_DATA_DIR / "test_data_raw.npy"
    tickers_path = RAW_DATA_DIR / "tickers.npy"
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)

    num_assets = 25
    train_days = 2011
    test_days = 503
    total_days = train_days + test_days 

    dummy_market_caps = np.random.uniform(low=100, high=1000, size=(total_days, num_assets))
    market_caps = dummy_market_caps # Dummy Market Caps for now 
    tickers = np.load(tickers_path)
    #Create Strategy
    arima = ArimaStrategy(preprocessors=[LogDiffByRow()])
    #Parameter grid is kept short for example purposes. Auto Arima parameter space is also kept small 
    #for the same reason.
    ml_params = {"hidden_layer_sizes":[(64,32)], "learning_rate":["constant"], "solver":["adam"],
    "window_size":[3,7]}
    ml = MLStrategy(models.mlp())
    arimaMlpHybrid = HybridStrategy(arima, ml, reuse=False) #reuse=False, training arima freshly
    #Execute it to get prediction results
    results = arimaMlpHybrid.run(train_data, test_data, ml_params, 20, 30, True)

    #Store results, model, model-figures, and model Statistics
    s = StorageManager()
    s.store_predictions(results, "model", "arimadiff-mlp")
    predictions_dict = {ticker:(true, pred) for ticker, true, pred in zip(tickers, test_data, results)}
    model_figure = MultiPlot("ArimaDiff-MLP")
    model_figure.plot(predictions_dict, cols=5)
    s.store_figures(model_figure, "model", "arimadiff-mlp")
    mse = mean_squared_error(results, test_data)
    s.store_statistics("model", "MSE", "arimadiff-mlp", mse)

    #Run Backtest, Save Backtest Results
    backtest = BackTesting(market_caps, train_data, test_data, results, 2.5, 200, 100)
    backtest_curve = backtest.equityCurve(10000)
    backtest_curve_dict = {"ArimaDiff-MLP":backtest_curve}
    backtest_plot = BacktestEquityPlot("ArimaDiff-MLP", "blue")
    backtest_plot.plot(backtest_curve_dict)
    s.store_predictions(backtest_curve, "portfolio", "arimadiff-mlp")
    s.store_figures(backtest_plot, "portfolio", "arimadiff-mlp")
    sharpe = Metrics.sharpe(backtest_curve)
    s.store_statistics("portfolio", "sharpe", "arimadiff-mlp", sharpe)








    

