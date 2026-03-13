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

