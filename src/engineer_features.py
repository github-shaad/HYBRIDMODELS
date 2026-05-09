import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import math
def smooth_exact_shape(data_matrix, window=5, method='ema'):
    df = pd.DataFrame(data_matrix.T)
    if method == 'ema':
        smoothed_df = df.ewm(span=window, adjust=False).mean()
    elif method == 'sma':
        smoothed_df = df.rolling(window=window, min_periods=1).mean()
    smoothed_df = smoothed_df.bfill().ffill()
    result_matrix = smoothed_df.values.T
    assert result_matrix.shape == data_matrix.shape, f"Shape mismatch! Expected {data_matrix.shape}, got {result_matrix.shape}"
    return result_matrix

class Features:
    def __init__(self, global_offset=0):
        self.global_offset = global_offset

    def generate_features(self, data, var_predictions, residuals, market_caps, lags, short_term, long_term):
        short_term_offset = self.global_offset - short_term
        medium_term = 60
        medium_term_offset = self.global_offset - medium_term
        long_term_offset = self.global_offset - long_term
        lag_offset = self.global_offset - lags

        # ----------------------------------------------------
        # 1. GENERATE FEATURES (Strictly using past data)
        # ----------------------------------------------------
        time_stamps = self.time_stamps_indicator(data)[:, lag_offset:]
        tickers = self.ticker_indicator(data)[:, lag_offset:]
        # Short term
        ema_l = self.ema_n(data, medium_term)[:, medium_term_offset:]
        sma_s = self.sma_n(data, short_term)[:, short_term_offset:]
        vol_s = self.vol_n(data, short_term)[:, short_term_offset:]
        positive_returns_s = self.positive_count_n(data, short_term)[:, short_term_offset:]
        
        # Long term
        sma_l = self.sma_n(data, long_term)[:, long_term_offset:]
        vol_l = self.vol_n(data, long_term)[:, long_term_offset:]
        z_score_s = self.rolling_z_score(data, short_term)[:, short_term_offset:]

        # Sector
        extra_returns_s = self.extra_returns_n(data, short_term)[:, short_term_offset:]
        vol_ratio_s = self.vol_ratio_n(data, short_term)[:, short_term_offset:]
        sector_vol_s = self.sector_vol_n(data, short_term)[:, short_term_offset:]
        

        mkp_lag_1 = self.lag_n(market_caps, lags)[:, lag_offset:]
        macd = ema_l - sma_l
        stripped_data = smooth_exact_shape(residuals[:, self.global_offset:], 3, 'ema') 
        stripped_var_preds = var_predictions[:, self.global_offset:]
        print(f"VAR PREDS SHAPE FEATURES{stripped_var_preds.shape}")
        # ----------------------------------------------------
        # 2. PRINT STATEMENTS FOR SHAPE AUDITING
        # ----------------------------------------------------
        """
        print("--- Feature Shape Audit ---")
        print(f"Input Data Shape:      {data.shape}")
        print(f"Time Stamps Shape:     {time_stamps.shape}")
        print(f"Tickers Shape:         {tickers.shape}")
        print(f"EMA Short Shape:       {ema_s.shape}")
        print(f"VOL Short Shape:       {vol_s.shape}")
        print(f"Pos Returns Shape:     {positive_returns_s.shape}")
        print(f"SMA Long Shape:        {sma_l.shape}")
        print(f"VOL Long Shape:        {vol_l.shape}")
        print(f"Z-Score Long Shape:    {z_score_l.shape}")
        print(f"Extra Returns Shape:   {extra_returns_s.shape}")
        print(f"Sector Vol Shape:      {sector_vol_s.shape}")
        print(f"Market Cap Lag Shape:  {mkp_lag_1.shape}")
        print(f"Target (Stripped) Shape: {stripped_data.shape}")
        print("---------------------------")
        """
        # ----------------------------------------------------
        # 3. STACK AND FLATTEN
        # ----------------------------------------------------
        # (Added positive_returns_s back into the stack!)
        features = np.stack(( stripped_var_preds,
            vol_s, 
            sma_s,
            z_score_s
        ), axis=-1)

        features_flat = features.reshape((-1, features.shape[2]))
        targets_flat = stripped_data.reshape(-1)
        return features_flat, targets_flat

    # ----------------------------------------------------
    # VECTORIZED HELPERS (Fast & Bug-Free)
    # ----------------------------------------------------
    def time_stamps_indicator(self, data):
        block = []
        for i, series in enumerate(data):
            indicator = np.cumsum(np.ones_like(series))[:-1] - self.global_offset + 1
            indicator = np.sin(indicator * 2 * math.pi / 7)
            block.append(np.trunc(indicator))
        return np.vstack(block)
    
    def ticker_indicator(self, data):
        block = []
        for i, series in enumerate(data):
            indicator = (np.ones_like(series)*(i+1))[:-1]
            block.append(np.trunc(indicator))
        return np.vstack(block)
    
    def lag_n(self, data, n):
        # Much cleaner vectorization than a loop
        return data[:, :-n]
    
    def sma_n(self, data, n):
        return np.mean(sliding_window_view(data, n, axis=1), axis=-1)[:, :-1]

    def ema_n(self, data, n):
        alpha = 2.0 / (n + 1)
        ema = np.empty_like(data, dtype=float)
        ema[:, :1] = data[:, :1]
        for t in range(1, data.shape[1]):
            ema[:, t:t+1] = alpha * data[:, t:t+1] + (1 - alpha) * ema[:, t-1:t]
        return ema[:, n-1:-1]
    
    def vol_n(self, data, n):
        return np.std(sliding_window_view(data, n, axis=1), axis=-1)[:, :-1]

    def rolling_z_score(self, data, n):
        sma_n = self.sma_n(data, n)
        vol_n = self.vol_n(data, n)
        # CRITICAL FIX: Align with yesterday's close to prevent look-ahead bias
        data_offset = data[:, n-1:-1] 
        return (data_offset - sma_n) / vol_n
    
    def sector_returns_n(self, data, n):
        windows = sliding_window_view(data, n, axis=1)
        day_sums = np.sum(windows, axis=(0, 2))
        return np.tile(day_sums, (data.shape[0], 1))[:, :-1]
    
    def extra_returns_n(self, data, n):
        daily_returns = n * self.sma_n(data, n)
        sector_returns = self.sector_returns_n(data, n)
        return daily_returns - sector_returns

    def sector_vol_n(self, data, n):
        windows = sliding_window_view(data, n, axis=1)
        vols = np.std(windows, axis=(0, 2))
        return np.tile(vols, (data.shape[0], 1))[:, :-1]

    def vol_ratio_n(self, data, n):
        daily_vol = self.vol_n(data, n)
        sector_vol = self.sector_vol_n(data, n)
        return daily_vol / sector_vol

    def positive_count_n(self, data, n):
        # CRITICAL FIX: Replaced logical hardcoded loop with robust boolean mask
        positive_mask = (data > 0).astype(int)
        counts = np.sum(sliding_window_view(positive_mask, n, axis=1), axis=-1)
        return counts[:, :-1]

