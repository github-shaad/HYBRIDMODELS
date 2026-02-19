from ensemble_factory import EnsembleFactory as ensem
from strategies import MLStrategy, HybridStrategy, ArimaStrategy
from preprocessors import DiffByRow, RowWiseMinMaxScaler, NormalizeByRow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from config.config import *
from plots import MultiPlot

train_data_path = RAW_DATA_DIR / "train_data_raw.npy"
test_data_path = RAW_DATA_DIR / "test_data_raw.npy"
a = np.load(train_data_path)
b = np.load(test_data_path)

arima_model = ArimaStrategy()

arima_res = arima_model.run(a, b, True)

svr_params = {"kernel":['rbf'], "C":[10, 100], "epsilon":[0.1], "window_size":[7]}
svr_model = MLStrategy(ensem.svr(), [RowWiseMinMaxScaler()])
hybrid_strat = HybridStrategy(arima_model, svr_model, True)

res = hybrid_strat.run(a, b, svr_params, test_size=50, jump_size=30, trace=True)

plot_dict = {"DUMMY": (b, res)}

svrPlot = MultiPlot("SVR test plot")

svrPlot.plot(plot_dict, 1, (15,10))
