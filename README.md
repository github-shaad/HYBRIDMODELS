# Hybrid VARMAX-ML trading strategies
This project aims to implement a number of  daily long short market neutral strategies aimed at the US telecommunications equity sector. We extend the framework proposed by [Zhang (2003)](https://doi.org/10.1016/S0925-2312(01)00702-0) by using a Vector AutoRegression with Exogenous Variables as the linear baseline and an ML algorithm as a non-linear correction term. We demonstrate that atleast one of the strategies generated consistent alpha uncorrelated with the S&P500 using Monte Carlo Permutation Tests, Skewness Checks, Rolling Metrics and other Robustness CHecks. We were able to generate a strategy/ies that is low volatility and is suitable for leveraged portfolios.

Running the strategy and getting results:
1. Run the config file once.
2. Run [ML]-train-and-test.py to automatically run for a certain VARMAX-[ML] strategy:
    1. The ML training and Validation Loop
    2. Signal Generation
    3. Backtesting 
3. Run additional checks with constant MODEL_NAME = "VARMAX-[ML]" for the additional checks described in the report.
4. Run Statistical significance tests(Monte Carlo Permutation tests) by running these three files:
    1. statistical-significance-hybridvsvar : For stat. sig. of Hybrid VARMAX-[ML] vs base VARX
    2. statistical-significance-sandp : For stat. sig. of VARMAX-[ML] / VARX against S&P500
    3. statistical-significance : For stat. sig. of VARMAX-[ML] / VARX against random signals
model_figures stores the true vs predicted time series for the test set and comparisons with the base VARMAX , while portfolio_figures stores equity-drawdown curves and comparisons with the S&P500 as well. raw_data stores all the raw data needed, model_predictions the predictions as a matrix of time series. Portfolio_predictions stores the equity curves for various strategies, saved_params saves the best model parameters. model_statistics.
