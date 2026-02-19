import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the FULL raw data (un-sliced) to get the anchor point correctly
full_test_data = np.load("test_data_raw.npy") 
print(full_test_data)
tickers = np.load("tickers.npy")

# 2. Load and Prepare Predictions
linear = np.load("outSamplePredictionsArimaDiff.npy")[:, 16:]
nonLinear = np.load("xgbResidualPredictionsDiff.npy")
nonLinear = np.reshape(nonLinear, (25, 492))[:, 5:] # Resulting shape: (25, 487)

# 3. Combine to get Total Predicted Log-Returns
total_log_returns = linear + nonLinear

# 4. Correct Inverse Transform Logic
# If your predictions represent the change starting at index 16, 
# the 'anchor' must be the price at index 15.
anchor_prices = full_test_data[:, 15:16] 

# The actual prices we are comparing against start at index 16
actual_prices = full_test_data[:, 16 : 16 + 487]

# Apply Cumulative Sum (to reverse differencing) and Exp (to reverse log)
cumulative_log_changes = np.cumsum(total_log_returns, axis=1)
predicted_prices = anchor_prices * np.exp(cumulative_log_changes)

print(f"Prediction Shape: {predicted_prices.shape}") # Should be (25, 487)
print(f"Actual Shape:     {actual_prices.shape}")    # Should be (25, 487)

# --- Plotting Code ---
num_tickers = predicted_prices.shape[0]
cols = 5
rows = (num_tickers // cols) + (1 if num_tickers % cols > 0 else 0)

fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
axes = axes.flatten()

for i in range(num_tickers):
    ax = axes[i]
    
    # Plot Actual Prices (Blue Dashed)
    ax.plot(actual_prices[i], color='#1f77b4', linestyle='--', alpha=0.8, label='Actual Price')
    
    # Plot Predicted Prices (Red Solid)
    ax.plot(predicted_prices[i], color='#d62728', linewidth=1.5, label='Predicted Price')
    
    ax.set_title(f"Ticker: {tickers[i]}", fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    
    if i == 0:
        ax.legend()

# Cleanup
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
fig.suptitle('Hybrid ARIMA-ANN: Reconstructed Price Levels', fontsize=16)
plt.show()

mse = mean_squared_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)
print(f"{mse}, {r2}")