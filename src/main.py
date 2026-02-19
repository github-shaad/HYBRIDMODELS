from ensemble_factory import EnsembleFactory as ensem
from strategies import MLStrategy, HybridStrategy, ArimaStrategy
from preprocessors import LogDiffByRow, RowWiseMinMaxScaler, NormalizeByRow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from config.config import *
from plots import MultiPlot, ResidPlot, MultiLineResidPlot, MultiLinePlot
from storageManager import StorageManager


train_data_path = RAW_DATA_DIR / "train_data_raw.npy"
test_data_path = RAW_DATA_DIR / "test_data_raw.npy"
a = np.load(train_data_path)[0:8]
b = np.load(test_data_path)[0:8]

arima_model = ArimaStrategy(preprocessors=[LogDiffByRow()])

res = arima_model.run(a, b, True)

print(f"True:{b}\nPreds:{res}")


r2 = r2_score(b, res)


s = StorageManager()
s.store_predictions(res, "model", "arima")
s.store_statistics("model", "Rsquared", "arima", r2)
