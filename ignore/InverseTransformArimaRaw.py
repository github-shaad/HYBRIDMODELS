import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
# --- 1. Load Data ---
# Load the full raw price data for comparison
full_test_data = np.load("test_data_raw.npy") 
tickers = np.load("tickers.npy")

# Load ARIMA predictions 
# These are already in price format (e.g., 150.25, 152.10)
# Slicing to match your window starting at index 16
predicted_prices = np.load("outSamplePredictionsArimaRaw.npy")[:, 16:]

# --- 2. Align Actuals ---
# Slice actual prices for comparison (index 16 onwards) to match prediction length
actual_prices = full_test_data[:, 16 : 16 + predicted_prices.shape[1]]

print(f"Prediction Matrix Shape: {predicted_prices.shape}")
print(f"Actual Matrix Shape:     {actual_prices.shape}")

# --- 3. Plotting ---
num_tickers = 25
cols, rows = 5, 5
fig, axes = plt.subplots(rows, cols, figsize=(20, 18))
axes = axes.flatten()

for i in range(num_tickers):
    ax = axes[i]
    
    # Plot Actual Prices
    ax.plot(actual_prices[i], color='#1f77b4', linestyle='--', alpha=0.7, label='Actual')
    
    # Plot Predicted Prices (Directly from ARIMA)
    ax.plot(predicted_prices[i], color='purple', linewidth=1.5, label='ARIMA (Levels)')
    
    ax.set_title(f"Ticker: {tickers[i]}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.94)
fig.suptitle('ARIMA Direct Price Level Predictions (No Preprocessing)', fontsize=16)
plt.show()
mse = mean_squared_error(actual_prices,predicted_prices)
r2 = r2_score(actual_prices,predicted_prices)
print(f"{mse}, {r2}")
