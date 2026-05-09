from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim


class ModelFactory:
    """
    Simple Wrapper Class for ML Regressors and ensembles as well.
    Passes Model params straight down and returns the respective model with those parameters.
    Regressons: xgBoost, MLPRegressor, etc..
    """
    @staticmethod
    def xgb(**params)->xgb.XGBRegressor:
        return xgb.XGBRegressor(n_estimators=150,           
            learning_rate=0.05,
            max_depth=3,                # Boosting depth is sequential, 3 is standard.
            subsample=0.7,              
            colsample_bytree=0.7,       
            reg_lambda=1.0,             
            random_state=42)
    
    @staticmethod
    def mlp(**params)->MLPRegressor:
        return MLPRegressor(hidden_layer_sizes=(32,),   # MATCHED with LSTM/GRU (1 layer of 32)
            max_iter=200,                # MATCHED with LSTM/GRU epochs
            early_stopping=False,       # MATCHED: Turned off because your custom PyTorch models don't have it
            alpha=0.1,                  
            learning_rate_init=0.001,   # MATCHED with PyTorch Adam LR
            random_state=42)
    
    @staticmethod
    def svr(**params):
        return SVR(kernel='rbf',               
            C=0.5,                      
            epsilon=0.01, 
            max_iter=200)
    
    @staticmethod
    def lstm(**params):
        return SklearnLSTM(**params)            
    
    @staticmethod
    def gru(**params):
        return SklearnGRU( hidden_dim=32,              # MATCHED with MLP/LSTM
            layers=1,                   # MATCHED with MLP/LSTM
            epochs=50,                  # MATCHED with MLP/LSTM
            lr=0.001,                   # MATCHED
            batch_size=32     # Matched LSTM for fair comparison
           )
    
    @staticmethod
    def StackingRegressor(models=None):
        return StackingRegressor(estimators=models)
    
    @staticmethod
    def rfr(**params):
        return RandomForestRegressor(n_estimators=150,           # MATCHED with XGBoost/GBR
            max_depth=5,                # NOTE: RF trees are independent, not sequential. They mathematically require slightly more depth than boosted trees to achieve the same capacity. Depth 5 here roughly equals Depth 3 in boosting.
            min_samples_leaf=10,        
            max_features='sqrt',        
            random_state=42)
    
    @staticmethod
    def gbr(**params):
        return GradientBoostingRegressor()
    
class _InternalLSTM(nn.Module):
    """
    _InternalLSTM: Internal LSTM class. Defines an LSTM object. For use by SkLearnLSTM
    """
    def __init__(self, input_dim, hidden_dim, layers, dropout=0.0):
        super(_InternalLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) 


class SklearnLSTM(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=1, hidden_dim=50, layers=1, 
                 epochs=50, lr=0.01, batch_size=32, dropout=0.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda")

    def fit(self, X, y):

        num_samples = X.shape[0]
   
        X_3d = X
        
        X_tensor = torch.tensor(X_3d, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        
   
        self.model = _InternalLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

  
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        actual_batch_size = min(self.batch_size, num_samples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            num_samples = X.shape[0]
            X_3d = X
            X_tensor = torch.tensor(X_3d, dtype=torch.float32).to(self.device)
            
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()   

    def set_params(self, **params):
        for key, value in params.items():
             setattr(self, key, value)
        return self             


class _InternalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout=0.0):
        super(_InternalGRU, self).__init__()
        
    
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        
     
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out)

class SklearnGRU(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=1, hidden_dim=50, layers=1, 
                 epochs=5, lr=0.01, batch_size=32, dropout=0.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        
        self.model = None
        self.criterion = None
        self.optimizer = None

    def fit(self, X, y):
 
        num_samples = X.shape[0]
        calculated_window_size = X.shape[1] // self.input_dim
        
      
        X_3d = X.reshape(num_samples, calculated_window_size, self.input_dim)
        
       
        X_tensor = torch.tensor(X_3d, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.model = _InternalGRU(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropout
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            
            num_samples = X.shape[0]
            calculated_window_size = X.shape[1] // self.input_dim
            
           
            X_3d = X.reshape(num_samples, calculated_window_size, self.input_dim)
            X_tensor = torch.tensor(X_3d, dtype=torch.float32)
            
            predictions = self.model(X_tensor)
            return predictions.numpy().flatten()       
        

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self                      


