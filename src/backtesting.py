import numpy as np
from portfolio import BlackLittermanPortfolio as Bl
from config.config import *

num_assets = 25
train_days = 2011
test_days = 503
total_days = train_days + test_days 

dummy_market_caps = np.random.uniform(low=100, high=1000, size=(total_days, num_assets))

class BackTesting:
    def __init__(self, market_caps, train_data, test_data, predictions, risk_averion, lookback_window, testing_length):
        market_caps_offset = np.shape(train_data)[1] - 1
        offset = np.shape(train_data)[1] - lookback_window
        self.market_caps = market_caps[market_caps_offset:,:]
        self.historical_prices = train_data[:, offset:]
        self.test_data = test_data
        self.predictions = predictions
        self.risk_aversion = risk_averion
        self.lookback_window = lookback_window
        self.testing_length = testing_length

    def pnlCurve(self):
        pass

    def get_returns(self, data, t):
        vector_t = data[:,t]
        vector_tminus = data[:,t-1]
        return (vector_t - vector_tminus) / vector_tminus

    def equityCurve(self, initial_value):
        daily_portfolio_value = [initial_value]
        past_data = self.historical_prices
        views = self.predictions[:,1]
        portfolio_value = initial_value

        for t in range(1, self.testing_length):
            weights_dict = Bl(self.market_caps[t-1], self.risk_aversion, past_data, views).get_portfolio()
            weights = np.array(list(weights_dict.values()))
            returns = self.get_returns(self.test_data, t)
            portfolio_returns = np.dot(weights, returns)
            portfolio_value = portfolio_value * (1+ portfolio_returns)
            daily_portfolio_value.append(float(portfolio_value))

            views = self.predictions[:,t]
            past_data = np.hstack((past_data[:,1:], self.test_data[:,t].reshape(-1,1)))
        
        return np.array(daily_portfolio_value)
            
    def mddCurve(self):
        pass


if __name__ == "__main__":
    train_path = RAW_DATA_DIR / "train_data_raw.npy"
    test_path = RAW_DATA_DIR / "test_data_raw.npy"
    pred_path = PREDICTIONS_DIR / "model_predictions" / "arima.npy"
    hist = np.load(train_path)
    preds = np.load(pred_path)
    test = np.load(test_path)
    
    tester = BackTesting(dummy_market_caps, hist, test, preds, 2.5, 100, 20)
    print(tester.equityCurve(10000))
