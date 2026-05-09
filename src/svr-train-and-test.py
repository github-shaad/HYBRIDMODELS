from model_factory import ModelFactory as models
from config.config import * 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import joblib
import json
from model_factory import ModelFactory as models
from strategies import VARStrategy
from engineer_features import Features
from numpy.lib.stride_tricks import sliding_window_view
from strategies import _Validation
from sklearn.model_selection import ParameterGrid
from plots import BacktestEquityDrawdownPlot
from metrics import Metrics
from backtesting import RankedBackTesting2
from storageManager import StorageManager, PlotSave
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler

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


TICKERS = np.load(RAW_DATA_DIR / "tickers.npy")[:10]
N_STOCKS = len(TICKERS)
MODEL_NAME = "VARMAX-SVR"
GLOBAL_OFFSET = 126
WINDOW_SIZE = 5
RECONSTRUCTION_OFFSET = GLOBAL_OFFSET + (WINDOW_SIZE - 1)
#DATA PREPROCESSING, CLEANING AND FEATURE SELECTION
data_path = RAW_DATA_DIR / "data.npy"
sandp500_path = RAW_DATA_DIR / "sandp.npy"
mkps_path = RAW_DATA_DIR / "market_caps.npy"
train_path  = RAW_DATA_DIR / "train_data_raw.npy"
test_path = RAW_DATA_DIR / "test_data_raw.npy"

data = np.load(data_path)
sandp = np.load(sandp500_path)
mkps = np.load(mkps_path).T


print(f"Loading Stage. Shapes of Loaded Data:")
print(f"Raw Price Data : {data.shape}")
print(f"Exogenous Data : {sandp.shape}")
print(f"Market caps : {mkps.shape}")

#size matched perfectly
raw_returns = (np.diff(data) / data[:,:-1]) 
raw_prices = data[:,:-1]
sandp_returns = np.diff(sandp) / sandp[:-1]
sandp_prices = sandp[:-1]
mkps_lag_1 = mkps[:,:-1]

#Smoothing
ewm_smoothed_raw_prices = np.log(smooth_exact_shape(raw_prices, 5, 'ema'))
ewm_smoothed_raw_returns = smooth_exact_shape(raw_returns, 5, 'ema')
#train_test_split
train_idx = int(np.floor(raw_returns.shape[1]*0.8))
smoothed_train_prices_log = ewm_smoothed_raw_prices[:,:train_idx]
train_prices_log = np.log(raw_prices[:,:train_idx])
test_prices_log = np.log(raw_prices[:,train_idx:])
train_mkps = mkps_lag_1[:,:train_idx]
test_mkps = mkps_lag_1[:,train_idx:]
sandp_prices = np.log(smooth_exact_shape(sandp_prices.reshape(-1,1))).flatten()
#model fit 
var_model = VARStrategy(15)
var_model.fit(smoothed_train_prices_log, sandp_prices[:train_idx], "True")
var_test_predicted_prices_log = var_model.get_predictions_test(test_prices_log, sandp_prices[:train_idx])
var_train_predicted_prices_log = var_model.get_predictions_train()

PlotSave().multiPlot(TICKERS, np.exp(test_prices_log), np.exp(var_test_predicted_prices_log), "VARMAX_TEST_1", "model")

train_residuals_svr = np.exp(train_prices_log) - np.exp(var_train_predicted_prices_log)

test_residuals_svr = np.hstack((train_residuals_svr[:,-GLOBAL_OFFSET:],np.exp(test_prices_log) - np.exp(var_test_predicted_prices_log)))
test_prices_log = np.hstack((train_prices_log[:,-GLOBAL_OFFSET:], test_prices_log))
test_mkps = np.hstack((train_mkps[:,-GLOBAL_OFFSET:], test_mkps))



param_grid = {
    # C: Penalty for error. Higher = tries to fit training data perfectly (risk of overfitting)
    'C': [50, 100],
    
    # epsilon: The "tube" where no penalty is given. 
    # Small = model is very strict. Large = model ignores small zags.
    'epsilon': [0.1, 0.5, 1, 5],
    
    # gamma: Reach of the kernel. 'scale' is a good default.
    'gamma': ['scale', 0.1, 0.01],
    
    # RBF is usually superior for non-linear stock residuals
    'kernel': ['rbf']
}



def create_sequence_for_lstm(data, var_preds, residuals, mkps, window_size):
    n_stocks = data.shape[0]
    features, _ = Features(GLOBAL_OFFSET).generate_features(data, var_preds, residuals, mkps, 1, 5, 40)
    n_features = features.shape[1]
    n_days = features.shape[0] // n_stocks
    features_3d = features.reshape((n_stocks, n_days, n_features))
    features_windows = sliding_window_view(features_3d, window_shape=window_size, axis=1)
    features_windows = features_windows.transpose(0,1,3,2)
    X = features_windows.reshape(-1, window_size, n_features)

    y = residuals[:,-n_days:][:,window_size-1:].reshape(-1)
    #X shape = Samples, time steps, features
    #Y shape = Stacked by Stocks
    return X, y

def validationLoopWalkForward(train_data, var_train_predictions, resid_train_data, mkps_train, val_days, start_day, jump_size, grid):
    print(f"Beginning Validation Loop")
    ids = _Validation.get_train_test(train_data, val_days, jump_size, start_day)
    parameter_grid = ParameterGrid(grid)
    print(f"Grid Size: {len(parameter_grid)}")
    best_params = None  
    best_score = -float('inf')
    for i , param in enumerate(parameter_grid):
        print(f"Parameter combination {i+1} of {len(parameter_grid)}")
        print(f"Testing Params : {param}")
        fold_scores = []
        for j, id in enumerate(ids):
            print(f"Training fold {j+1} : train index {id[0]} validation index {id[1]}")

            train_fold = train_data[:,:id[0]]
            resid_train_fold = resid_train_data[:,:id[0]]
            var_train_predictions_fold = var_train_predictions[:,:id[0]]
            mkp_train_fold = mkps_train[:,:id[0]]

            val_fold = train_data[:,id[0] - GLOBAL_OFFSET:id[1]]
            resid_val_fold = resid_train_data[:,id[0] - GLOBAL_OFFSET:id[1]]
            var_val_predictions_fold = var_train_predictions[:,id[0] - GLOBAL_OFFSET:id[1]]
            mkp_val_fold = mkps_train[:,id[0] - GLOBAL_OFFSET:id[1]]

            train_X, train_y = Features(GLOBAL_OFFSET).generate_features(train_fold, var_train_predictions_fold,
                                                              resid_train_fold, mkp_train_fold, 1, 5, GLOBAL_OFFSET)
            val_X, val_y = Features(GLOBAL_OFFSET).generate_features(val_fold, var_val_predictions_fold, 
                                                          resid_val_fold, mkp_val_fold, 1, 5, GLOBAL_OFFSET)
            
            model = models.svr(**param)
            scaler_x = RobustScaler()
            scaler_y = RobustScaler()
            train_X = scaler_x.fit_transform(train_X)
            train_y = scaler_y.fit_transform(train_y.reshape(-1,1)).flatten()
            model.fit(train_X, train_y)
            val_preds_y = model.predict(val_X)
            val_preds_y = scaler_y.inverse_transform(val_preds_y.reshape(-1,1)).flatten()
            score = r2_score(val_fold[:,GLOBAL_OFFSET:], val_preds_y.reshape(N_STOCKS, -1) + var_val_predictions_fold[:,GLOBAL_OFFSET:])
            fold_scores.append(score)
            print(f"Score: {score}")
        avg_score = np.mean(fold_scores)
        print(f"Average Fold Score : {avg_score}")
        if avg_score > best_score:
            best_score = avg_score
            best_params = param
    print(f"Best Score : {best_score}")
    print(f"Best Params : {best_params}")
    print("Walk Forward Validation Finished")
    print("Training Model on Full Train Set")
    model.set_params(**best_params)
    train_X, train_y = Features(GLOBAL_OFFSET).generate_features(train_data, var_train_predictions, resid_train_data, mkps_train, 1, 5, GLOBAL_OFFSET)
    train_X = scaler_x.fit_transform(train_X)
    train_y = scaler_y.fit_transform(train_y.reshape(-1,1)).flatten()
    model.fit(train_X, train_y)
    print("Done!!")
    return best_params, model, scaler_x, scaler_y

best_params, model, scaler_x, scaler_y = validationLoopWalkForward(np.exp(train_prices_log), np.exp(var_train_predicted_prices_log),
                                                train_residuals_svr, train_mkps, 252, 500, 252, param_grid)

test_X, test_y = Features(GLOBAL_OFFSET).generate_features(np.exp(test_prices_log), np.hstack((np.exp(var_train_predicted_prices_log)[:,-GLOBAL_OFFSET:],np.exp(var_test_predicted_prices_log))),
                                                test_residuals_svr, test_mkps, 1, 5, GLOBAL_OFFSET)

test_X = scaler_x.transform(test_X)
test_y = scaler_y.transform(test_y.reshape(-1,1)).flatten()
svr_predictions = model.predict(test_X)

test_y = scaler_y.inverse_transform(test_y.reshape(-1,1)).flatten()
svr_predictions = scaler_y.inverse_transform(svr_predictions.reshape(-1,1)).flatten()
svr_true_2d = test_y.reshape(N_STOCKS, -1)
svr_predictions_2d = svr_predictions.reshape(N_STOCKS, -1)



true_2d = np.exp(test_prices_log)[:,GLOBAL_OFFSET:]
corrected_preds_2d = np.exp(var_test_predicted_prices_log) + svr_predictions_2d


#MODEL METRICS, PLOTS
StorageManager.store_predictions(corrected_preds_2d, "model", MODEL_NAME)
StorageManager.store_model(model, "hybrid", MODEL_NAME)
StorageManager.store_params(best_params, MODEL_NAME)
Metrics.save_model_metrics(MODEL_NAME, true_2d, corrected_preds_2d)
PlotSave.multiPlot(TICKERS, true_2d, corrected_preds_2d, MODEL_NAME, "model")
PlotSave.tripleMultiPlot(TICKERS, true_2d, np.exp(var_test_predicted_prices_log), corrected_preds_2d, 
                         MODEL_NAME, "model")


#Backtesting

def generate_signals(true_prices, predicted_prices):
    """
    Converts predicted prices into an expected return signal matrix.
    Both inputs should be shape (N_STOCKS, Time)
    """
    today_price = true_prices[:, :-1]
    tomorrow_predicted = predicted_prices[:, 1:]
    expected_returns = (tomorrow_predicted - today_price) / today_price
    return expected_returns, today_price


signal_matrix, tradable_prices = generate_signals(true_2d, corrected_preds_2d)

backtest = RankedBackTesting2(signal_matrix, tradable_prices, 3,3, 0.001)
equity_curve, drawdown_curve = backtest.equityDrawdownCurve(10_000)
Metrics.save_portfolio_metrics(MODEL_NAME, equity_curve, backtest.get_weights())
btPlot = BacktestEquityDrawdownPlot(MODEL_NAME)
btPlot.plot({MODEL_NAME:(equity_curve, drawdown_curve)})
StorageManager.store_figures(btPlot, "portfolio", f"{MODEL_NAME}_ranked")
StorageManager.store_predictions(equity_curve, "portfolio", MODEL_NAME)


