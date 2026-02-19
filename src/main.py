from ensemble_factory import EnsembleFactory as ensem
from strategies import MLStrategy, HybridStrategy, ArimaStrategy
from preprocessors import DiffByRow, RowWiseMinMaxScaler, NormalizeByRow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from config.config import *

train_data_path = RAW_DATA_DIR / "train_data_raw.npy"
a = np.load(train_data_path)
print("worked")

b = np.load("test_data_raw.npy")

arima_model = ArimaStrategy()

arima_res = arima_model.run(a, b, True)



svr_params = {"kernel":['rbf'], "C":[10, 100], "epsilon":[0.1, 0.5], "window_size":[7, 14, 21]}
svr_model = MLStrategy(ensem.svr(), [RowWiseMinMaxScaler()])
hybrid_strat = HybridStrategy(arima_model, svr_model, True)

res = hybrid_strat.run(a, b, svr_params, test_size=50, jump_size=30, trace=True)
