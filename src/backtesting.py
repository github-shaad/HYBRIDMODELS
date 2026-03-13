import numpy as np
from portfolio import BlackLittermanPortfolio as Bl
from config.config import *



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
            
    def equityDrawdownCurve(self, initial_value):
        equity_curve = self.equityCurve(initial_value)
        drawdown_curve = []
        
        for i in range(np.shape(equity_curve)[0]):
            peak = np.max(equity_curve[:i+1])
            drawdown = (equity_curve[i] - peak) / peak
            drawdown_curve.append(drawdown)

        return np.column_stack((equity_curve, np.array(drawdown_curve)))
  


if __name__ == "__main__":
    pass
