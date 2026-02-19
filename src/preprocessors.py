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

class DiffByRow(BaseProcessor):
    def __init__(self, anchor):
        pass

    def transform(self, data):
        chunk = []
        epsilon = 1e-8
        for e in data:
            safe_e = np.where(e <= 0, epsilon, e) 
            row = np.log(safe_e)
            row_diff = [row[0]] + [row[j] - row[j-1] for j in range(1, len(row))]
            chunk.append(row_diff)
        return np.vstack(chunk)

    def inverse_transform(self, predictions, x_original=None):
        if x_original is None:
         
            anchors = predictions[:, 0]
            diffs = predictions[:, 1:]
            chunk = []
            for i, row in enumerate(diffs):
                reconstructed_row = []
  
                reconstructed_row.append(np.exp(anchors[i])) 
                
                curr_log_price = anchors[i]
                for diff in row:
                    curr_log_price += diff
                    reconstructed_row.append(np.exp(curr_log_price))
                chunk.append(reconstructed_row)
            return np.vstack(chunk)

     
        is_full_reconstruction = (predictions.shape == x_original.shape)

        if is_full_reconstruction:
             predictions_to_use = predictions[:, 1:]
        else:
             predictions_to_use = predictions

    
        offset = len(x_original) - len(predictions)
        if offset > 0:
            anchors = x_original[offset-1 : -1, -1] 
            if len(anchors) != len(predictions): anchors = x_original[offset-1 : -1]
        else:
            anchors = x_original[:, 0]

        chunk = []
        for i, row in enumerate(predictions_to_use):
            reconstructed_row = []
            
        
            curr_log_price = np.log(anchors[i] if anchors[i] > 0 else 1e-8)
            
    
            if is_full_reconstruction:
                reconstructed_row.append(np.exp(curr_log_price))

   
            for j in range(len(row)):
                curr_log_price += row[j]
                reconstructed_row.append(np.exp(curr_log_price))
            
            chunk.append(reconstructed_row)
            
        return np.vstack(chunk)

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


