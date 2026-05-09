import numpy as np
from config.config import *
import numpy as np
from config.config import *

class RankedBackTesting2:
    def __init__(self, signals, prices, longs=2, shorts=2, threshold=0):
        self.signals = signals
        self.prices = prices
        self.longs = longs
        self.shorts = shorts
        self.n_stocks, self.time_steps = signals.shape
        self.portfolio_history = []
        self.daily_returns = []
        self.trade_log = []
        self.threshold = threshold
        self.weights_history = []
    
    def equityCurve(self, capital):
        curr_capital = capital
        self.portfolio_history.append(curr_capital)

        for t in range(self.time_steps - 1):
            daily_signal = self.signals[:,t]
            actual_returns = (self.prices[:, t+1] - self.prices[:, t]) / self.prices[:, t]
            ranked_indices = np.argsort(daily_signal)
            candidate_shorts = ranked_indices[:self.shorts]
            candidate_longs = ranked_indices[-self.longs:]

            short_idx = candidate_shorts[daily_signal[candidate_shorts] < -self.threshold]
            long_idx = candidate_longs[daily_signal[candidate_longs] > self.threshold]

            # NEW: Initialize today's weight array (all zeros)
            daily_weights = np.zeros(self.n_stocks)

            # Assign weights based on how many stocks passed the threshold
            if len(long_idx) > 0:
                long_weight_per_stock = 0.5 / len(long_idx)
                daily_weights[long_idx] = long_weight_per_stock
                
            if len(short_idx) > 0:
                short_weight_per_stock = -0.5 / len(short_idx)
                daily_weights[short_idx] = short_weight_per_stock

            
            self.weights_history.append(daily_weights)

            long_return = np.mean(actual_returns[long_idx]) if len(long_idx) > 0 else 0.0
            short_return = np.mean(-actual_returns[short_idx]) if len(short_idx) > 0 else 0.0

            portfolio_return = (0.5 * long_return) + (0.5 * short_return)
            curr_capital = curr_capital * (1 + portfolio_return)
            
            self.portfolio_history.append(curr_capital)
            self.daily_returns.append(portfolio_return)
        return np.array(self.portfolio_history)
        
    def equityDrawdownCurve(self, initial_value):
        equity_curve = self.equityCurve(initial_value)
        
        # Calculate drawdown efficiently in O(N) instead of O(N^2)
        running_max = np.maximum.accumulate(equity_curve)
        
        # Prevent division by zero if starting value was 0 (unlikely, but safe)
        running_max = np.where(running_max == 0, 1e-8, running_max)
        
        drawdown_curve = (equity_curve - running_max) / running_max

        return equity_curve, drawdown_curve
    
    def get_weights(self):
        return np.array(self.weights_history)
    
if __name__ == "__main__":
    pass
