from ensemble_factory import EnsembleFactory as ensem
from strategies import MLStrategy, HybridStrategy, ArimaStrategy
from preprocessors import LogDiffByRow, RowWiseMinMaxScaler, NormalizeByRow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from config.config import *
from plots import MultiPlot, ResidPlot, MultiLineResidPlot, MultiLinePlot

train_data_path = RAW_DATA_DIR / "train_data_raw.npy"
test_data_path = RAW_DATA_DIR / "test_data_raw.npy"
a = np.load(train_data_path)[0:4]
b = np.load(test_data_path)[0:4]

arima_model = ArimaStrategy(preprocessors=[LogDiffByRow()])

res = arima_model.run(a, b, True)

print(f"True:{b}\nPreds:{res}")
plot_dict = {"DUMMY": (b[0], res[0]), "DUMMY 2": (b[1], res[1]), "DUMMY 3": (b[2], res[2]), "DUMMY 4": (b[3], res[3])}

p = MultiPlot("test")

q = MultiLinePlot("test residuals", ["green", "blue", "black", "yellow"])
q.plot(b[0], plot_dict, figsize=(16,9))

