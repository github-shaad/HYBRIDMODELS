import preprocessors as pspr
import numpy as np
from numpy.typing import NDArray
import pmdarima as pm
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.api import VAR
from engineer_features import Features
from sklearn.decomposition import PCA
from metrics import Metrics
class VARStrategy:
    """
    VARStrategy: Class that defines a multivariate Vector Autoregression Strategy.
    Acts as a drop-in replacement for ArimaStrategy.
    """
    def __init__(self, maxlags=15, information_criterion='aic', preprocessors=None):
        self.spec = "var"
        self.maxlags = maxlags
        self.information_criterion = information_criterion
        self.fittedModel = None
        self.lag_order = 0
        self.train_data_T = None # Stores transposed train data for forecasting
        self.preprocessors = preprocessors # Assuming pspr is imported in your script
        self._pca = None
    def fit_preprocessor(self, data):
        self.preprocessors.fit_transform(data)

    def preprocess(self, data):
        return self.preprocessors.transform(data)
    
    def inv_preprocess(self, data):
        return self.preprocessors.inverse_transform(data)

    def fit(self, data, exog, trace=False):
        if trace:
            print("Begin VAR Fitting\n--------------\n")
            
        # 1. Transpose data to (n_time_steps, n_assets) for statsmodels

        self.train_data_T = data.T
        self._pca = PCA(n_components=4)
        train_components = self._pca.fit_transform(self.train_data_T)
        # 2. Fit the VAR model
        model = VAR(train_components, exog=exog)
        self.fittedModel = model.fit(maxlags=15, ic='aic')
        
        # Check if it stubbornly chose 0 lags
        if self.fittedModel.k_ar == 0:
            print("VAR chose 0 lags. Forcing a lag of 1 to maintain baseline structure.")
            # Refit the model explicitly forcing 1 lag (no 'ic' parameter)
            self.fittedModel = model.fit(1)
            self.lag_order = 1
        else:
            self.lag_order = self.fittedModel.k_ar
            print(f"VAR selected lag order: {self.lag_order}")
        self.lag_order = self.fittedModel.k_ar
        
        if trace:
            print(f"VAR automatically selected a lag order of: {self.lag_order}")
            print("Process finished with no errors")

    def get_predictions_train(self):
        # Extract fitted values
        preds_T = self.fittedModel.fittedvalues
        
        # VAR drops the first 'lag_order' steps. We pad with zeros 
        # to ensure the output shape exactly matches the original train_data shape.
        pad_length = self.train_data_T.shape[0] - preds_T.shape[0]
        if pad_length > 0:
            padding = np.zeros((pad_length, preds_T.shape[1]))
            preds_T = np.vstack((padding, preds_T))
            
        # Transpose back to (n_assets, n_time_steps)
        self.predictions_train = preds_T
        return self._pca.inverse_transform(self.predictions_train).T
    
    def get_raw_residuals_train(self, train_data):
        return train_data - self.inv_preprocess(self.get_predictions_train())
    def get_raw_residuals_test(self,test_data, exog_train):
        return test_data - self.inv_preprocess(self.get_predictions_test(test_data, exog_train))
    
    def get_predictions_test(self, test_data, exog):
        steps = test_data.shape[1]
        #exog forecast
        exog_future_model = pm.auto_arima(exog,stationary=True,d=0,seasonal=False,max_p=3,max_q=3,                 
                                        information_criterion='bic', trace=True,suppress_warnings=True )
        
        exog_future = exog_future_model.predict(n_periods=test_data.shape[1])
        # VAR requires the last 'lag_order' days of the training set to kick off the forecast
        last_known_data = self._pca.transform(self.train_data_T[-self.lag_order:])
        
        # Forecast the exact length of the test data
        preds_T = self.fittedModel.forecast(last_known_data, steps=steps, exog_future=exog_future)
        
        # Transpose back to (n_assets, n_time_steps)
        return self._pca.inverse_transform(preds_T).T

    
    def get_residuals_test(self, test_data, mkps_test):
        return test_data - self.get_predictions_test(test_data, mkps_test)
    
    def run(self, train_data, test_data, exog_train, trace=False):
        self.fit_preprocessor(train_data)
        train_data_preprocessed = self.preprocess(train_data)
        self.preprocessors.test_reference_prices = test_data
        test_data_preprocessed = self.preprocess(test_data)
        
        self.fit(train_data_preprocessed, exog_train, trace)
        
        test_predictions = self.get_predictions_test(test_data_preprocessed, exog_train)
        test_predictions_raw = self.inv_preprocess(test_predictions)

        return test_predictions_raw
    
    def set_params(self, **params):
        """
        Updates internal parameters dynamically.
        Example: var.set_params(maxlags=10, information_criterion='bic')
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: VARStrategy has no parameter '{key}'. Ignoring.")
        
        self.fittedModel = None 
        return self

class ArimaStrategy:
    def __init__(self, start_p=0, max_p=2, start_q=0, max_q=2, seasonal=False,information_criterion='bic', preprocessors=None):
        self.spec = "arima"
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.seasonal = seasonal
        self.information_criterion = information_criterion
        self.fittedModels = []
        self.preprocessors = preprocessors

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
                maxiter=50,
                stepwise=True,
                d=1,
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
        self.preprocess.test_reference_data = test_data
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
        self.offset_for_arima = 60

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

    def fit_final(self, X, y):
        self.model.fit(X, y)

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
            if trace:
                print(f"Parameter combination {i+1} of {len(grid)}")
            window_size = params["window_size"]
            ml_params = {k: v for k, v in params.items() if k != "window_size"}
            fold_score = []

            if trace:
                print(f"Testing Parameters:{params}")

            for i, id in enumerate(ids):
                if trace:
                    print(f"Walk Forward fold {i+1} of {len(ids)}")
                train_fold = train_data[:, :id[0]]
                val_fold = train_data[:, id[0]:id[1]]

                self.model.set_params(**ml_params)

                self.fit(train_fold, window_size)

                predictions_fold = self.predict_rolling(train_fold, val_fold)

                score = r2_score(val_fold, predictions_fold)
                if trace:
                    print(f"Score: {score}")
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

    def best_fit(self, train_X, train_y, params):
        grid = ParameterGrid(params)
        print("Finding Best Fit")
        best_score = -float("inf")
        best_params = None
        for i, params in enumerate(grid):
            print(f"parameter combination {i+1} of {len(grid)} -> {params}\n----------")
            self.model.set_params(**params)
            self.fit_final(train_X, train_y)
            res = self.model.predict(train_X)
            score = r2_score(train_y, res)
            print(f"Score : {score}")
            if score > best_score:
                best_score = score
                best_params = params
        print(f"Best Score: {best_score}")
        self.best_params = best_params
        self.model.set_params(**self.best_params)
        self.fit_final(train_X, train_y)

    def best_fit_walk_forward_final(self, train_data, mkps, params, test_size, jump_size, isPrices=True):
        ids = _Validation.get_train_test(train_data, test_size, jump_size)
        grid = ParameterGrid(params)
        best_score = -float("inf")
        best_params = None
        print("Starting Best Fit Walk Forward")
        for i, params in enumerate(grid):
            print(f"Parameter combination {i+1} of {len(grid)}-> {params}\n----------")
            fold_score = []
            for i, id in enumerate(ids):
                print(f"fold {i+1} of {len(ids)}")
                if i == 0:
                    train_fold = train_data[:, :id[0]]
                else:
                    train_fold = train_data[:, ids[i-1][0]:id[0]]
                val_fold = train_data[:, id[0]:id[1]]
                feature_target_pair_train = Features(60).generate_features(train_fold, mkps, 1, 5, 60)
                train_fold_X = feature_target_pair_train[0]
                train_fold_y = feature_target_pair_train[1]

                feature_target_pair_val = Features(60).generate_features(val_fold, mkps, 1, 5, 60)
                val_fold_X = feature_target_pair_val[0]
                val_fold_y = feature_target_pair_val[1]
                self.model.set_params(**params)
                self.fit_final(train_fold_X, train_fold_y)
                val_preds = self.model.predict(val_fold_X)
                score = r2_score(val_fold_y, val_preds)
                score = Metrics.calculate_hit_rate(val_fold_y, val_preds, isPrices)
                fold_score.append(score)
            avg_score = np.mean(fold_score)
            print(f"Average hit rate = {avg_score}")
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
            print("----------")
        print(f"Best hit rate = {best_score}")
        print(f"Best Params = {best_params}")
        best_model = self.model.set_params(**best_params)
        self.model = best_model
        self.best_params = best_params
        feature_target = Features(60).generate_features(train_data, mkps, 1, 5, 60)
        train_X = feature_target[0]
        train_y = feature_target[1]
        self.fit_final(train_X, train_y)

    def predict_recursive(self, train_data, n_steps):
        window = np.shape(train_data)[1] - self.window_size
        curr_batch = train_data[:, window:]
        future_predictions = []
        for i in range(n_steps):
            preds = self.model.predict(curr_batch).reshape(-1,1)
            future_predictions.append(preds)
            curr_batch = np.hstack((curr_batch[:,1:], preds))

        return np.hstack(future_predictions)
    
    def predict_rolling(self, train_data, test_data):
        """
        Performs a rolling 1-step forecast. 
        Instead of feeding predictions back into the model, it uses the 
        actual ground-truth data from the test set as it moves forward.
        """
        # Combine train and test so we can easily slice rolling windows
        full_data = np.hstack((train_data, test_data))
        n_train = np.shape(train_data)[1]
        n_test = np.shape(test_data)[1]
        
        future_predictions = []
        
        # Step through the test data one day at a time
        for i in range(n_test):
            # Grab the actual historical window strictly BEFORE the target day
            start_idx = n_train + i - self.window_size
            end_idx = n_train + i
            
            curr_batch = full_data[:, start_idx:end_idx]
            
            # Predict exactly 1 step ahead using ACTUAL history
            preds = self.model.predict(curr_batch).reshape(-1, 1)
            future_predictions.append(preds)
            
        return np.hstack(future_predictions)

    def run(self, train_data, test_data, parameter_grid, test_size=50, jump_size=30, trace=False):
        self.fit_preprocessor(train_data)
        train_data_preprocessed = self.preprocess(train_data)
        test_data_preprocessed = self.preprocess(test_data)
        self.best_fit_walk_forward(train_data_preprocessed, parameter_grid, test_size, jump_size, trace)
        predictions = self.predict_rolling(train_data_preprocessed, test_data_preprocessed)
        predictions_raw = self.inv_preprocess(predictions)

        return predictions_raw

    def run_final(self, train_X, train_y, test_X, param_grid, test_size=50, jump_size=30, isPrices=True):
        self.best_fit_walk_forward_final_feature_target(train_X, train_y, param_grid, test_size,
                                                        jump_size, isPrices)
        preds = self.model.predict(test_X)
        return preds.reshape((25,-1))
    
    def best_fit_walk_forward_final_feature_target(self, train_X, train_y, 
                                                   params, test_size, jump_size, isPrices=True):
        ids = _Validation_feature_target.get_train_test(train_X, test_size,
                                                        jump_size, n_assets=25)
        grid = ParameterGrid(params)
        best_params = None
        best_score = -float('inf')
        print("Starting Best Fit Walk Forward")
        for i, params in enumerate(grid):
            print(f"Parameter combination {i+1} of {len(grid)}-> {params}\n----------")
            fold_score = []
            for j, id in enumerate(ids):
                print(f"fold {j+1} of {len(ids)}")
                
                train_fold_X = train_X[ : id[0], : ]
                train_fold_y = train_y[ : id[0]]

                val_fold_X = train_X[ id[0] : id[1], : ]
                val_fold_y = train_y[ id[0] : id[1]]

                self.model.set_params(**params)
                self.fit_final(train_fold_X, train_fold_y)
                val_preds = self.model.predict(val_fold_X)

                score = r2_score(val_fold_y, val_preds)
                fold_score.append(score)


            avg_score = np.mean(fold_score)
            print(f"Average R2 = {avg_score}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

            print("----------")


        print(f"Best R2 = {best_score}")
        print(f"Best params = {best_params}")
        best_model = self.model.set_params(**best_params)
        self.model = best_model
        self.best_params = best_params
        self.fit_final(train_X, train_y)

class HybridStrategy:
    def __init__(self, arima_model, ml_model, reuse=False):
        self.arima_model:VARStrategy= arima_model
        self.ml_model:MLStrategy = ml_model
        self.reuse = reuse
        self.residual_true = None
        self.residual_pred = None

    def run_final(self, train_data, test_data, train_X, train_y, test_X, test_y,
                  exog_train, params):
        self.arima_model.fit_preprocessor(train_data)
        train_data_p = self.arima_model.preprocess(train_data)
        test_data_p = self.arima_model.preprocess(test_data)
        self.arima_model.fit(train_data, exog_train)
        self.residual_true = self.arima_model.get_raw_residuals_train(train_data_p)
        self.residual_pred = self.arima_model.get_raw_residuals_test(test_data_p, exog_train)

        ml_preds = self.ml_model.run_final()

       

    def run(self, train_data, test_data, params, test_size, jump_size, trace=False):
        self.arima_model.fit_preprocessor(train_data)
        train_data_preprocessed = self.arima_model.preprocess(train_data)
        self.arima_model.preprocessors.test_reference_prices = test_data
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
        self.residual_true = test_data - self.arima_model.inv_preprocess(arima_preds)
        self.residual_pred = test_data - final_predictions_raw

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



class _Validation_feature_target:
    @staticmethod
    def _get_length(data):
        """
        Dynamically determines the number of time-steps/samples 
        regardless of whether raw data or a feature matrix is passed.
        """
        # If passed a tuple like (X, y), grab the first element
        if isinstance(data, tuple) or isinstance(data, list):
            data = data[0]
            
        shape = np.shape(data)
        
        # 1D array (e.g., flattened targets)
        if len(shape) == 1:
            return shape[0]
        # 2D feature matrix (samples/time down the rows, features across columns)
        elif shape[0] > shape[1]:
            return shape[0]
        # Old raw data format (tickers down the rows, time across columns)
        else:
            return shape[1]

    @staticmethod
    def _get_id(data, test_size=1, jump_size=1):
        n = _Validation_feature_target._get_length(data)
        n_red = n - test_size
        i = n_red % jump_size
        return i

    @staticmethod
    def get_train_test(data, test_size, jump_size, min_train_id=30, n_assets=25):
        test_rows = test_size*n_assets
        jump_rows = jump_size*n_assets
        min_train_rows = min_train_id*n_assets
        ids = []
        n = _Validation_feature_target._get_length(data)
        n_red = n - test_rows
        t = test_rows
        initial = _Validation_feature_target._get_id(data, test_rows, jump_rows)
        
        while initial + min_train_rows <= n_red:
            ids.append([initial + min_train_rows, initial + t + min_train_rows])
            initial += jump_rows
            
        return np.vstack(ids) if ids else np.array([])

        