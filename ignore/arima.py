import pmdarima as pm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import numpy as np


train_data = np.load("train_data_diff.npy")
test_data  = np.load("test_data_diff.npy")
print(f"Train shape: {train_data.shape}")
print(f"Test shape:  {test_data.shape}")


def train(train_data,test_data):
    in_sample_residuals = []
    in_sample_predictions = []
    out_sample_predictions = []
    out_sample_residuals = []
    i = 1
    for train_series, test_series in zip(train_data, test_data):
        print(f"Training on series: {train_series} {i}")
        fittedModel = pm.auto_arima(train_series, seasonal=False, d=None, stepwise=False, test='adf',trace=False, error_action='ignore', suppress_warnings=True, start_p=1, max_p=5, start_q=1, max_q=5)
        in_sample_forecast = fittedModel.predict_in_sample()
        in_sample_predictions.append(in_sample_forecast)
        out_sample_forecast = fittedModel.predict(n_periods=test_data.shape[1])
        out_sample_predictions.append(out_sample_forecast)        
        in_sample_residue = [float(actual - pred) for actual, pred in zip(train_series, in_sample_forecast)]
        in_sample_residuals.append(in_sample_residue)
        out_sample_residue = [float(actual - pred) for actual, pred in zip(test_series, out_sample_forecast)]
        out_sample_residuals.append(out_sample_residue)
        i+=1
    return in_sample_predictions, out_sample_predictions, in_sample_residuals, out_sample_residuals


results = train(train_data, test_data)

in_sample_predictions = np.array(results[0])
out_sample_predictions = np.array(results[1])
in_sample_residuals = np.array(results[2])
out_sample_residuals = np.array(results[3])

np.save("inSamplePredictionsArimaDiff.npy", in_sample_predictions)
np.save("outSamplePredictionsArimaDiff.npy", out_sample_predictions)
np.save("inSampleResidualsDiff.npy", in_sample_residuals)
np.save("outSampleResidualsDiff.npy", out_sample_residuals)


