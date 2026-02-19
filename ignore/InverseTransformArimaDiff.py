import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Load Data ---
# Load the full raw price data to get the anchor point
full_test_data = np.load("test_data_raw.npy") 
tickers = np.load("tickers.npy")

# Load ARIMA predictions (assumed to be Log-Returns)
# Slicing to match your window (e.g., 487 steps)
arima_log_returns = np.load("outSamplePredictionsArimaDiff.npy")[:, 16:]

# --- 2. Inverse Transform Logic ---
# The 'anchor' is the actual price at the step immediately before the forecast.
# If predictions start at index 16, the anchor is index 15.
anchor_prices = full_test_data[:, 15:16] 

# Reconstruct Prices:
# Cumulative Sum (undoes differencing) -> Exp (undoes log) -> Multiply by Anchor
cumulative_log_changes = np.cumsum(arima_log_returns, axis=1)
predicted_prices_arima = anchor_prices * np.exp(cumulative_log_changes)

# Slice the actual prices to match the prediction window for comparison
actual_prices = full_test_data[:, 16 : 16 + arima_log_returns.shape[1]]

# --- 3. Plotting the Results ---
num_tickers = 25
cols = 5
rows = 5
fig, axes = plt.subplots(rows, cols, figsize=(20, 18))
axes = axes.flatten()

for i in range(num_tickers):
    ax = axes[i]
    
    # Plot Actual Prices
    ax.plot(actual_prices[i], color='#1f77b4', linestyle='--', alpha=0.7, label='Actual')
    
    # Plot ARIMA-only Predicted Prices
    ax.plot(predicted_prices_arima[i], color='orange', linewidth=1.5, label='ARIMA Forecast')
    
    ax.set_title(f"{tickers[i]}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.94)
fig.suptitle('ARIMA Price Level Reconstruction (No ANN)', fontsize=16)
plt.show()

mse = mean_squared_error(actual_prices,predicted_prices_arima )
r2 = r2_score(actual_prices,predicted_prices_arima )
print(f"{mse}, {r2}")