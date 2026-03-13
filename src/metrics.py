import numpy as np
class Metrics:
    @staticmethod
    def sharpe(returns, risk_free_rate=0.0):
        daily_returns = np.diff(returns) / returns[:-1]

        mean_returns = np.mean(daily_returns)
        std_returns = np.std(daily_returns)

        if std_returns == 0: 
            return 0
        
        daily_risk_free_rate = (1+risk_free_rate)**(1/252) - 1

        daily_sharpe = (mean_returns - daily_risk_free_rate)/std_returns

        annualized_ratio = daily_sharpe * np.sqrt(252)

        return annualized_ratio 


