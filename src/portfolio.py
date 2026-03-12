import numpy as np
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, black_litterman
from config.config import *



class BlackLittermanPortfolio:
    def __init__(self, weights, risk_aversion, historical_prices, predictions):
        self.weights = np.array(weights) 
        self.risk_aversion = risk_aversion
        self.historical_prices = historical_prices.T 
        self.predictions = predictions

    def get_portfolio(self):

        Sigma = risk_models.sample_cov(self.historical_prices, returns_data=False)

        delta = self.risk_aversion 

        prior_returns = black_litterman.market_implied_prior_returns(self.weights, delta, Sigma)

        Q = self.predictions
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
    weights = BlackLittermanPortfolio([0.4, 0.5, 0.1], 2.2, hist, preds).get_portfolio()
    print(weights)