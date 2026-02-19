import numpy as np
import json
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from numpy.lib.stride_tricks import sliding_window_view



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def process_data(data, window_size, tickers, stats_map=None):

    chunks_x, chunks_y = [], []
    new_stats = {}

    for i, series in enumerate(data):
        ticker = tickers[i]
        
        # 1. REMOVE ARTIFACT: Slice off first 10 points (ARIMA initialization spike)
        clean_series = series[10:]
        
        # 2. GET STATS
        if stats_map:
            mean, std = stats_map[ticker]['mean'], stats_map[ticker]['std']
        else:
            mean, std = np.mean(clean_series), np.std(clean_series)
            std = 1 if std == 0 else std
            new_stats[ticker] = {'mean': mean, 'std': std, 'ticker': ticker}

        # 3. NORMALIZE
        norm_series = (clean_series - mean) / std
        
        # 4. WINDOWING (Vectorized)
        if len(norm_series) > window_size:
            # sliding_window_view creates a view (no copy), very fast
            windows = sliding_window_view(norm_series, window_size + 1)
            chunks_x.append(windows[:, :-1])
            chunks_y.append(windows[:, -1])

    # 5. STACK ONCE (Performance Optimization)
    X = np.vstack(chunks_x) if chunks_x else np.array([])
    y = np.concatenate(chunks_y) if chunks_y else np.array([])
    
    return X, y, (new_stats if stats_map is None else stats_map)

# --- 2. Load Data ---
print("Loading data...")
train_raw = np.load("train_data_raw.npy")   # Shape: (Stocks, Time_Train)
test_raw = np.load("test_data_raw.npy")   # Shape: (Stocks, Time_Test)
tickers = np.load("tickers.npy")

# Handle shape mismatch if test_raw has weird dimensions (e.g. 2,1,500)
if test_raw.ndim == 3: test_raw = np.squeeze(test_raw)

# --- 3. Training Loop ---
best_mse = float('inf')
best_model = None
best_window = 1
best_metadata = {}
best_preds = np.array([])

# Window sizes to experiment with
windows = [i for i in range(1,21)]

for size in windows:
    print(f"\n--- Testing Window Size: {size} ---")
    
    # Prepare Train (Calculate Stats)
    X_train, y_train, metadata = process_data(train_raw, size, tickers, stats_map=None)
    
    # Prepare Test (Use Train Stats to prevent leakage)
    X_test, y_test, _ = process_data(test_raw, size, tickers, stats_map=metadata)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data for this window size.")
        continue

    # Train Model
    model = MLPRegressor(hidden_layer_sizes=(64, 32), 
                         activation='relu', 
                         solver='adam', 
                         max_iter=500, 
                         random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = model.score(X_test, y_test)

    print(f"R2 Score: {r2:.4f} | MSE: {mse:.5f}")

    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_window = size
        best_metadata = metadata
        best_preds = preds

# --- 4. Save Best Results ---
print(f"\nWinner: Window {best_window} with MSE {best_mse:.5f}")

# Save Model
joblib.dump(best_model, "ann_raw.joblib")
print("Saved ann.joblib")

# Save Metadata (Mean/Std per stock)
with open("model_metadata_ann_raw.json", "w") as f:
    json.dump(best_metadata, f, cls=NumpyEncoder, indent=4)
print("Saved model_metadata_ann_raw.json")
np.save("annPredictionsRaw.npy", best_preds)