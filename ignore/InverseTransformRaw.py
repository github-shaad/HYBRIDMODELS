import numpy as np 
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

raw_data = np.load("test_data_raw.npy")[:, 18:]
linear = np.load("outSamplePredictionsArimaRaw.npy")[:, 18:]
nonLinear = np.load("xgbResidualPredictionsRaw.npy")
tickers = np.load("tickers.npy")
nonLinear = np.reshape(nonLinear, (25, 485))
print(np.shape(linear))
print(np.shape(nonLinear))
final = linear + nonLinear

fig, axs = plt.subplots(5,5, figsize=(50,50))
axs = axs.flatten()

for i in range(len(axs)):
    ticker = tickers[i]
    axs[i].plot(final[i], color='red', alpha=0.5, linewidth=0.8, label='Predicted')
    axs[i].plot(raw_data[i], color='blue',linestyle="--", linewidth=0.8, label='Actual')
    axs[i].set_title(f"{ticker}", fontsize=10, fontweight='bold')
    axs[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.96) # Make room for the main title
plt.show()

mse = mean_squared_error(final, raw_data)
r2 = r2_score(raw_data, final)
print(f"{mse}, {r2}")