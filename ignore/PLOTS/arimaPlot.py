import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

raw_data = np.load("raw_series.npy")
outsample = np.load("outSamplePredictionsArimaDiff.npy")
test_data = np.load("test_data_diff.npy")
in_sample_residuals = np.load("inSampleResidualsDiff.npy")
out_sample_residuals = np.load("outSampleResidualsDiff.npy")
tickers = np.load("tickers.npy")

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(50, 50))
axes = axes.flatten()
# Loop through and plot each on its own axis
for i in range(25):
    axes[i].plot(outsample[i], color='red')
    axes[i].plot(test_data[i], color="blue", alpha=0.5)
    axes[i].set_title(f"Stock: {tickers[i]}")
    axes[i].grid(True)

plt.tight_layout(pad=3.0)
plt.show() 