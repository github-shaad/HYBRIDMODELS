import numpy as np
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, black_litterman
from config.config import *



class BlackLitterman:
    def __init__(self, weights, risk_aversion, historical_prices, predictions_matrix):
        # Good practice: explicitly cast weights to a numpy array
        self.weights = np.array(weights) 
        self.risk_aversion = risk_aversion
        
        # Transpose historical prices so shape is (Timesteps, Assets) for sample_cov
        self.historical_prices = historical_prices.T 
        
        # DO NOT transpose this. Keep it as (Assets, Timesteps)
        self.predictions_matrix = predictions_matrix 

    def get_portfolio(self):
        # Note: If 'hist' contains raw prices (e.g., $150), returns_data should be False.
        # If 'hist' already contains % returns (e.g., 0.05), returns_data=True is correct.
        Sigma = risk_models.sample_cov(self.historical_prices, returns_data=True)

        # Use the risk aversion passed to the class
        delta = self.risk_aversion 

        prior_returns = black_litterman.market_implied_prior_returns(self.weights, delta, Sigma)
         
        # Slices the last column (the latest timestep) for all 3 assets
        Q = self.predictions_matrix[:, -1]
        
        # P must be 3x3. We use the length of the weights array to get the asset count.
        P = np.eye(len(self.weights))

        bl = BlackLittermanModel(Sigma, pi=prior_returns, Q=Q, P=P)

        ret_bl = bl.bl_returns()
        Sigma_bl = bl.bl_cov()

        ef = EfficientFrontier(ret_bl, Sigma_bl)
        weights = ef.max_sharpe()
        clean_weights = ef.clean_weights()

        return clean_weights
    

if __name__ == "__main__":
    train_path = RAW_DATA_DIR / "train_data_raw.npy"
    pred_path = PREDICTIONS_DIR / "model_predictions" / "arima.npy"
    hist = np.load(train_path)[:3,:20]
    preds = np.load(pred_path)[:3,:20]
    weights = BlackLitterman([0.4, 0.5, 0.1], 2.2, hist, preds).get_portfolio()
    print(weights)