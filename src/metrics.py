import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from storageManager import StorageManager
class Metrics:
    @staticmethod
    def risk_aversion(returns, risk_free_rate=.043):
        theta = (np.mean(returns)*252 - risk_free_rate) / (np.var(returns)*252)
        return theta if theta >= 1 else 1
    
    @staticmethod
    def sharpe(equity_curve, risk_free_rate=0.0):
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]

        mean_returns = np.mean(daily_returns)
        std_returns = np.std(daily_returns)

        if std_returns == 0: 
            return 0
        
        daily_risk_free_rate = (1+risk_free_rate)**(1/252) - 1

        daily_sharpe = (mean_returns - daily_risk_free_rate)/std_returns

        annualized_ratio = daily_sharpe * np.sqrt(252)

        return annualized_ratio 

    @staticmethod
    def cagr(equity_curve):
        total_days = len(equity_curve)
        total_return = equity_curve[-1]/equity_curve[0]
        cagr = (total_return ** (252/total_days)) - 1

        return float(cagr)
    
    @staticmethod
    def mdd(equity_curve):
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak

        return float(np.min(drawdowns))
    
    @staticmethod
    def sortino(equity_curve, annual_risk_free_rate=0.0):
        """Calculates the annualized Sortino Ratio (penalizes only downside risk)."""
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Isolate only the negative returns for downside deviation
        negative_returns = daily_returns[daily_returns < 0]
        
        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return 0.0 # Avoid division by zero
            
        mean_return = np.mean(daily_returns)
        std_downside = np.std(negative_returns)
        daily_rf = (1 + annual_risk_free_rate)**(1/252) - 1
        
        daily_sortino = (mean_return - daily_rf) / std_downside
        return float(daily_sortino * np.sqrt(252))

    @staticmethod
    def calmar(equity_curve):
        mdd = Metrics.mdd(equity_curve)
        cagr = Metrics.cagr(equity_curve)

        if mdd == 0:
            return 0
        
        return float(cagr/abs(mdd))
    
    @staticmethod
    def turnover(weight_matrix=None):
        if weight_matrix == None:
            return 0
        weight_changes = np.diff(weight_matrix, axis=0)
        
        abs_weight_changes = np.abs(weight_changes)
        daily_turnover = np.sum(abs_weight_changes, axis=1) / 2.0
    
        return np.mean(daily_turnover) 
    @staticmethod
    def calculate_hit_rate(actual, predictions, is_prices=True):
        """
        Calculates the Directional Accuracy (Hit Rate) of time series predictions.
        
        Args:
            actual (array-like): The true values from your test set.
            predictions (array-like): The predicted values from your model.
            is_prices (bool): Set to True if inputs are absolute prices. 
                            Set to False if inputs are already daily returns.
                            
        Returns:
            float: The percentage of correctly predicted directions (0.0 to 1.0).
        """
        actual = np.array(actual)
        predictions = np.array(predictions)
        
        if is_prices:
            if actual.ndim == 2:
                # Actual market direction: Actual price today - Actual price yesterday
                actual_direction = np.sign(actual[:,1:] - actual[:,:-1])
                
                # Model's traded direction: Predicted price today - Actual price yesterday
                predicted_direction = np.sign(predictions[:,1:] - actual[:,:-1])
            else:
                # Actual market direction: Actual price today - Actual price yesterday
                actual_direction = np.sign(actual[1:] - actual[:-1])
                
                # Model's traded direction: Predicted price today - Actual price yesterday
                predicted_direction = np.sign(predictions[1:] - actual[:-1])                
        else:
            # If you pass returns directly, we just check if they are positive or negative
            actual_direction = np.sign(actual)
            predicted_direction = np.sign(predictions)
            
        # Compare the two arrays to see where the signs match
        correct_guesses = (actual_direction == predicted_direction)
        
        # Calculate the percentage of correct guesses
        hit_rate = np.mean(correct_guesses)
        
        return hit_rate
    
    @staticmethod
    def directional_rsquared(actual, predictions, isPrices=True):
        h = Metrics.calculate_hit_rate(actual, predictions, isPrices)
        rsq = r2_score(actual, predictions)
        x = h * np.exp(rsq)
        return x
    @staticmethod
    def save_model_metrics(model_name, true, pred):
        s = StorageManager()
        s.store_statistics("model", "Rsquared", model_name, r2_score(true,pred))
        s.store_statistics("model", "MAE", model_name, mean_absolute_error(true, pred))
        s.store_statistics("model", "MSE", model_name, mean_squared_error(true, pred))
        s.store_statistics("model", "MAPE", model_name, mean_absolute_percentage_error(true, pred))
        s.store_statistics("model", "Hit Rate", model_name, Metrics.calculate_hit_rate(true,pred))
    
    @staticmethod
    def save_portfolio_metrics(model_name, equity_curve, weight_matrix=None):
        s = StorageManager()
        s.store_statistics("portfolio", "sharpe", model_name, Metrics.sharpe(equity_curve))
        s.store_statistics("portfolio", "sortino", model_name, Metrics.sortino(equity_curve))
        s.store_statistics("portfolio", "cagr", model_name, Metrics.cagr(equity_curve))
        s.store_statistics("portfolio", "mdd", model_name, Metrics.mdd(equity_curve))
        s.store_statistics("portfolio", "calmar", model_name, Metrics.calmar(equity_curve))
        s.store_statistics("portfolio", "turnover", model_name, Metrics.turnover(weight_matrix))