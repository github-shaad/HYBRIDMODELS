from model_factory import ModelFactory as ensem
from strategies import MLStrategy, HybridStrategy, ArimaStrategy
from preprocessors import LogDiffByRow, RowWiseMinMaxScaler, NormalizeByRow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from config.config import *
from plots import MultiPlot, ResidPlot, MultiLineResidPlot, MultiLinePlot
from storageManager import StorageManager

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

    s = StorageManager()
    arimaPlots = MultiPlot("Arima")
    arimaPlots.plot(data_dict, cols=5)
    s.store_figures(arimaPlots, "model", "arima_raw_data")