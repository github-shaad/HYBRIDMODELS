"""
Preprocessors
"""
from abc import ABC, abstractmethod
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np


class BaseProcessor(ABC):

    @abstractmethod
    def fit_transform(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass

class PipeLine:
    def __init__(self, steps):
        if steps == None:
            self.steps = []
        else:
            self.steps = steps

    def fit_transform(self, data):
        x = data.copy()
        for step in self.steps:
            step.fit_transform(data)

    def transform(self, data):
        x = data.copy()
        for step in self.steps:
            x = step.transform(x)
        return x

    def inverse_transform(self, data):
        x = data.copy()
        for step in reversed(self.steps):
            x = step.inverse_transform(x)    
        return x    

class Naive(BaseProcessor):
    def __init__(self):
        pass

    def fit_transform(self, data):
        return data
    
    def transform(self, data):
        return data
    
    def inverse_transform(self, data):
        return data


class NormalizeByRow(BaseProcessor):
    """
    Row Wise Normalization. \n
    f = NormalizeByRow : A_{mn} -> B_{mn} such that f([A_{m1}, A_{m2}, ....]) = [(A_{mn} - mu)/std, ....]
    """
    def __init__(self):
        self.mean = []
        self.std = []
    
    def fit_transform(self, data):
        for series in data:
            self.mean.append(np.mean(series))
            self.std.append(np.std(series))
    
    def transform(self, data):
        res = []
        for i, series in enumerate(data):
            res.append((series - self.mean[i])/self.std[i])

        return np.vstack(res)

    def inverse_transform(self, data):
        res = []
        for i, series in enumerate(data):
            res.append((series*self.std[i]) + self.mean[i])
        return np.vstack(res)


class LogDiffByRow(BaseProcessor):
    def __init__(self):
        # We store the LAST price of the training set to start the Test set
        self.anchors = [] 

    def fit_transform(self, data):
        # Fit stores the last known price of each row in training
        self.anchors = [series[-1] for series in data]
        return self.transform(data)
    
    def transform(self, data):
        res = []
        for series in data:
            # We use np.diff on the logs
            log_series = np.log(series)
            # Prepend 0 to keep the same shape, or handle first element
            diffs = np.zeros_like(log_series)
            diffs[1:] = np.diff(log_series)
            res.append(diffs)
        return np.vstack(res)
    
    def inverse_transform(self, data):
        res = []
        for j, diff_series in enumerate(data):
            # To get back, we need the cumulative sum of log returns
            # starting from the log of our anchor
            log_anchor = np.log(self.anchors[j])
            
            # The reconstructed log prices are: log_anchor + cumsum(diffs)
            reconstructed_log = log_anchor + np.cumsum(diff_series)
            res.append(np.exp(reconstructed_log))
            
        return np.vstack(res)           
      
class LogDiffByRowSpecial(BaseProcessor): # Assuming it inherits from BaseProcessor
    def __init__(self):
        self.anchors = [] 
        self.test_reference_prices = None # <-- THE BACKDOOR

    def fit_transform(self, data):
        self.anchors = [series[-1] for series in data]
        return self.transform(data)
    
    def transform(self, data):
        res = []
        for series in data:
            log_series = np.log(series)
            diffs = np.zeros_like(log_series)
            diffs[1:] = np.diff(log_series)
            res.append(diffs)
        return np.vstack(res)
    
    def inverse_transform(self, data):
        res = []
        for j, diff_series in enumerate(data):
            # If we provided the actual test data (Backtesting Mode)
            if self.test_reference_prices is not None:
                # Build the array of ACTUAL yesterday prices: [Train Anchor, Test_Day_0, Test_Day_1, ...]
                actual_prev_prices = np.concatenate(([self.anchors[j]], self.test_reference_prices[j, :-1]))
                
                # Add today's predicted return to yesterday's ACTUAL price (NO CUMSUM)
                reconstructed_log = np.log(actual_prev_prices) + diff_series
                res.append(np.exp(reconstructed_log))
                
            # If we are flying blind into the future (Live Trading Mode)
            else:
                log_anchor = np.log(self.anchors[j])
                reconstructed_log = log_anchor + np.cumsum(diff_series)
                res.append(np.exp(reconstructed_log))
                
        return np.vstack(res)
    

class RowWiseMinMaxScaler(BaseProcessor):
    def __init__(self, factor=[0,1]):
        self.infinima = factor[0]
        self.suprema = factor[1]
        self.maximum = []
        self.minimum = []

    def fit_transform(self, data):
        for i, series in enumerate(data):
            self.maximum.append(max(series))
            self.minimum.append(min(series))
    
    def transform(self, data):
        x = []
        for i, series in enumerate(data):
            series_new = self.infinima + ((series - self.minimum[i])*(self.suprema - self.infinima)/(self.maximum[i] - self.minimum[i]))
            x.append(series_new)

        return np.vstack(x)    

    def inverse_transform(self, data):
        x = []
        for i, series in enumerate(data):
            series_new = self.minimum[i] + ((series - self.infinima)*(self.maximum[i] - self.minimum[i])/(self.suprema - self.infinima))
            x.append(series_new)

        return np.vstack(x)
import numpy as np

class DiffByRow(BaseProcessor): 
    def __init__(self):
        self.anchor_prices = None

    def fit_transform(self, data):
        # Save ONLY the final column of the known data to anchor future predictions
        self.anchor_prices = data[:, -1]
        return self.transform(data)
    
    def transform(self, data):
        # Calculate returns: (P_t - P_{t-1}) / P_{t-1}
        returns = np.diff(data, axis=1) / data[:, :-1]
        
        # Pad the first column with zeros to maintain the exact same shape as input
        zero_padding = np.zeros((data.shape[0], 1))
        padded_returns = np.hstack((zero_padding, returns))
        
        return padded_returns
    
    def inverse_transform(self, predicted_returns):
        """
        Converts predicted future returns back into raw prices, 
        springboarding off the last known prices from fit_transform.
        """
        if self.anchor_prices is None:
            raise ValueError("Anchor prices not found. Run fit_transform first.")
            
        # P_t = P_anchor * cumulative_product(1 + R_t)
        cum_returns = np.cumprod(1 + predicted_returns, axis=1)
        projected_prices = self.anchor_prices[:, None] * cum_returns
        
        return projected_prices

class Cleaner:
    @staticmethod
    def clip(data, lower_percentile, upper_percentile):
        lower_bound = np.percentile(data, lower_percentile, axis=0)
        upper_bound = np.percentile(data, upper_percentile, axis=0)
        return np.clip(data, lower_bound, upper_bound)
    
    
class SlidingWindowNormalize(BaseProcessor):
    def __init__(self, windowSize):
        pass

class SlidingWindow(BaseProcessor):
    def __init__(self, windowSize):
        self.windowSize = windowSize

    def fit_transform(self, data):
        return super().fit_transform(data)
    
    def transform(self, data):
        x = np.array([])
        chunk = []
        for i, e in enumerate(data):
            a = sliding_window_view(data[i], self.windowSize)
            chunk.append(a)

        x = np.vstack(chunk)   
        return x    

    def inverse_transform(self, data):
        return super().inverse_transform(data)

