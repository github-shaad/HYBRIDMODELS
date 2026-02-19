import preprocessors as pspr
import numpy as np
from numpy.typing import NDArray
import pmdarima as pm
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid


class ArimaStrategy:
    def __init__(self, start_p=1, max_p=5, start_q=1, max_q=5, seasonal=False,information_criterion='aic', preprocessors=None):
        self.spec = "arima"
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.seasonal = seasonal
        self.information_criterion = information_criterion
        self.fittedModels = []
        self.preprocessors = pspr.PipeLine(preprocessors)

    def fit_preprocessor(self, data):
        self.preprocessors.fit_transform(data)

    def preprocess(self, data):
        return self.preprocessors.transform(data)
    
    def inv_preprocess(self, data):
        return self.preprocessors.inverse_transform(data)

    def fit(self, data, trace):
        self.fittedModels = []

        if trace:
            print("Begin Auto Arima Fitting\n--------------\n")
        for i, series in enumerate(data):
            if trace:
                print(f"Time Series:{i+1}")
            self.fittedModels.append(pm.auto_arima(series, start_p=self.start_p, max_p=self.max_p,
                start_q=self.start_q, max_q=self.max_q,
                seasonal=self.seasonal,
                information_criterion=self.information_criterion,
                error_action='ignore',
                suppress_warnings=True,
                test="adf",
                trace=False))
        if trace:
            print("Process finished with no errors")

    def get_predictions_train(self):
        predictions = np.array([])
        chunk = []
        for model in self.fittedModels:
            chunk.append(model.predict_in_sample())

        predictions = np.vstack(chunk)
        self.predictions_train = predictions
        return predictions
    
    def get_predictions_test(self, test_data):
        predictions_test = np.array([])
        chunk = []
        for model in self.fittedModels:
            chunk.append(model.predict(n_periods=test_data.shape[1]))

        predictions_test = np.vstack(chunk) 

        return predictions_test

    def get_residuals_train(self):
        chunk = []
        for model in self.fittedModels:
            chunk.append(model.resid())
        dataMatrix = np.vstack(chunk)

        return dataMatrix
    
    def get_residuals_test(self, test_data):
        return test_data - self.get_predictions_test(test_data)
    
    def run(self, train_data, test_data, trace=False):
        self.fit_preprocessor(train_data)
        train_data_preprocessed = self.preprocess(train_data)
        test_data_preprocessed = self.preprocess(test_data)
        self.fit(train_data_preprocessed, trace)
        test_predictions = self.get_predictions_test(test_data_preprocessed)
        test_predictions_raw = self.inv_preprocess(test_predictions)

        return test_predictions_raw

    
    def set_params(self, **params):
        """
        Updates internal parameters dynamically.
        Example: arima.set_params(max_p=7, seasonal=True)
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: ArimaStrategy has no parameter '{key}'. Ignoring.")
        
        self.fittedModels = [] 
        return self
    

class MLStrategy:
    """
    MLStrategy: Class that defines an ML-only Strategy.
    """
    def __init__(self, MLStrat, preprocessors=None):
        self.preprocessors = pspr.PipeLine(preprocessors)
        self.model = MLStrat
        self.window_size = 0
        self.best_params = {}
    def fit_preprocessor(self, data):
        self.preprocessors.fit_transform(data)

    def preprocess(self, data):    
        return self.preprocessors.transform(data)
    
    def inv_preprocess(self,data):
        return self.preprocessors.inverse_transform(data)     

    def feature_target_pair(self, train_data, window_size):
        self.window_size = window_size
        data = self.sliding_window_transform(train_data, self.window_size)
        x = data[:,:-1]
        y = data[:,-1]
        return (x,y)
    
    def shorten_test_data(self, test_data):
        return test_data[:, self.window_size-1:]
    
    def sliding_window_transform(self, data, windowSize):
        x = pspr.SlidingWindow(windowSize+1).transform(data)    
        return x  
        
    def fit(self, train_data, windowSize):
        self.window_size = windowSize
        pair = self.feature_target_pair(train_data, self.window_size)
        train_x = pair[0]
        train_y = pair[1]
        self.model = self.model.fit(train_x, train_y)

    def best_fit_walk_forward(self, train_data:NDArray, params:dict, test_size:int, jump_size:int, trace:bool)->None:
        """
        Performs Walk-Forward Validation (TimeSeriesSplit) to find best params.
        """
        if trace:
            print(f"Shape of Training set : {np.shape(train_data)}\n---------------------------\nBeginning Custom Walk Forward Validation")
        ids = _Validation.get_train_test(train_data, test_size, jump_size)
        
        grid = ParameterGrid(params)

        best_score = -float('inf')
        best_params = None
        best_window = None
        for i, params in enumerate(grid):
            window_size = params["window_size"]
            ml_params = {k: v for k, v in params.items() if k != "window_size"}
            fold_score = []

            if trace:
                print(f"Testing Parameters:{params}")

            for id in ids:
                train_fold = train_data[:, :id[0]]
                val_fold = train_data[:, id[0]:id[1]]

                self.model.set_params(**ml_params)

                self.fit(train_fold, window_size)

                predictions_fold = self.predict_recursive(train_fold, np.shape(val_fold)[1])

                score = r2_score(val_fold, predictions_fold)
                fold_score.append(score)
            
            average_score = np.mean(fold_score)

            if trace:
                print(f"Average score : {average_score}")
            if average_score > best_score:
                best_score = average_score
                best_params = params   
                best_window = window_size    
        if trace:
            print(f"\nBest Parameters:{best_params}\nBest Average Score:{best_score}")

        best_params = {k:v for k, v in best_params.items() if k != "window_size"}
        best_model = self.model.set_params(**best_params)
        self.model = best_model
        self.window_size = best_window
        self.best_params = best_params
        self.fit(train_data, best_window)

    def predict_recursive(self, train_data, n_steps):
        window = np.shape(train_data)[1] - self.window_size
        curr_batch = train_data[:, window:]
        future_predictions = []
        for i in range(n_steps):
            preds = self.model.predict(curr_batch).reshape(-1,1)
            future_predictions.append(preds)
            curr_batch = np.hstack((curr_batch[:,1:], preds))

        return np.hstack(future_predictions)

    def run(self, train_data, test_data, parameter_grid, test_size=50, jump_size=30, trace=False):
        self.fit_preprocessor(train_data)
        train_data_preprocessed = self.preprocess(train_data)
        test_data_preprocessed = self.preprocess(test_data)
        self.best_fit_walk_forward(train_data_preprocessed, parameter_grid, test_size, jump_size, trace)
        predictions = self.predict_recursive(train_data_preprocessed, np.shape(test_data_preprocessed)[1])
        predictions_raw = self.inv_preprocess(predictions)

        return predictions_raw



class HybridStrategy:
    def __init__(self, arima_model, ml_model, reuse=False):
        self.arima_model:ArimaStrategy = arima_model
        self.ml_model:MLStrategy = ml_model
        self.reuse = reuse

    def run(self, train_data, test_data, params, test_size, jump_size, trace=False):
        self.arima_model.fit_preprocessor(train_data)
        train_data_preprocessed = self.arima_model.preprocess(train_data)
        test_data_preprocessed = self.arima_model.preprocess(test_data)

        if not self.reuse:
            self.arima_model.fit(train_data_preprocessed, trace)

        arima_preds = self.arima_model.get_predictions_test(test_data_preprocessed)

        train_residuals = self.arima_model.get_residuals_train()
        test_residuals = self.arima_model.get_residuals_test(test_data_preprocessed)

        # New training and test set for ml. I can simply call ml best fit walk forward and itll be done

        ml_preds = self.ml_model.run(train_residuals, test_residuals, params, test_size, jump_size, trace)

        final_predictions = arima_preds + ml_preds
        final_predictions_raw = self.arima_model.inv_preprocess(final_predictions)

        return final_predictions_raw














        
    
        

                



                



class _Validation:
    @staticmethod
    def _get_id(data, test_size=1, jump_size=1):
        n = np.shape(data)[1]
        n_red = n - test_size
        i = n_red % jump_size
        return i

    @staticmethod
    def get_train_test(data, test_size, jump_size, min_train_id=30):
        ids = []
        n_red = np.shape(data)[1] - test_size
        t = test_size
        initial = _Validation._get_id(data, test_size, jump_size)
        while initial + min_train_id <= n_red:
            ids.append([initial+min_train_id, initial+t+min_train_id])
            initial+=jump_size
        return np.vstack(ids)
        

        