import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import math
from numpy.lib.stride_tricks import sliding_window_view

# --- 1. Load Data & Model ---
print("Loading files...")
try:
    metadata = json.load(open("model_metadata_ann_raw.json", "r"))
    model = joblib.load("ann_raw.joblib")
    tickers = np.load("tickers.npy")
    test_data = np.load("test_data_raw.npy") # The actual ground truth
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Handle shape issues (squeeze (25, 1, 503) -> (25, 503))
if test_data.ndim == 3:
    test_data = np.squeeze(test_data)

# AUTO-DETECT WINDOW SIZE
# The model knows exactly how many inputs it needs (e.g., 15)
WINDOW_SIZE = 1
print(f"Loaded Model. Expects Window Size: {WINDOW_SIZE}")
print(f"Test Data Shape: {test_data.shape}")

# --- 2. Prediction Helper Function ---
def get_stock_prediction(ticker_idx, window_size):
    """
    Returns (Actual, Predicted) for a stock by index.
    Returns (None, None) if data is invalid.
    """
    ticker = tickers[ticker_idx]
    
    # Check for metadata
    if ticker not in metadata:
        return None, None
        
    # Get raw series & Clean Artifacts (slice off first 10)
    raw_series = test_data[ticker_idx]
    clean_series = raw_series[10:]
    
    # Get stats from Training Metadata
    stats = metadata[ticker]
    mean, std = stats['mean'], stats['std']

    # Normalize
    norm_series = (clean_series - mean) / std

    # Windowing
    # If series is too short for the window, skip it
    if len(norm_series) <= window_size:
        return None, None
        
    windows = sliding_window_view(norm_series, window_size + 1)
    
    X_input = windows[:, :-1]
    y_true_norm = windows[:, -1]

    # Predict & Denormalize
    pred_norm = model.predict(X_input)
    
    pred_real = (pred_norm * std) + mean
    actual_real = (y_true_norm * std) + mean
    
    return actual_real, pred_real

# --- 3. Plotting Loop ---
num_stocks = len(tickers)
cols = 5  # How many charts per row?
rows = math.ceil(num_stocks / cols)

# Create a massive figure (adjust height based on rows)
fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten() # Flatten so we can loop easily (axes[0], axes[1]...)

print(f"Generating plots for {num_stocks} stocks...")

for i in range(len(axes)):
    # If we have more subplots than stocks, hide the extras
    if i >= num_stocks:
        axes[i].axis('off')
        continue

    # Get Data
    ticker = tickers[i]
    actual, predicted = get_stock_prediction(i, WINDOW_SIZE)

    # Plot
    if actual is not None:
        axes[i].plot(actual, color='blue', alpha=0.5, linewidth=0.8, label='Actual')
        axes[i].plot(predicted, color='red', linestyle='--', linewidth=1.0, label='Pred')
        
        axes[i].set_title(f"{ticker}", fontsize=10, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Only add legend to the first plot to keep it clean
        if i == 0:
            axes[i].legend(fontsize='small')
    else:
        axes[i].text(0.5, 0.5, "Insufficient Data", ha='center', va='center')

# --- 4. Final Layout Adjustments ---
plt.tight_layout()
plt.subplots_adjust(top=0.96) # Make room for the main title
plt.suptitle(f"ANN Residual Predictions (Window Size: {WINDOW_SIZE})", fontsize=16)

plt.show()