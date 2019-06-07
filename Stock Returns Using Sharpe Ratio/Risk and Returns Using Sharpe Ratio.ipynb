# Analyzing Stocks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock_data = pd.read_csv('stock_data.csv', parse_dates = ['Date'], 
                         index_col = ['Date'], na_values = 'NAN')
benchmark_data = pd.read_csv('benchmark_data.csv', parse_dates = ['Date'], 
                             index_col = ['Date'], na_values = 'NAN')

stock_data.isnull().sum()
benchmark_data.isnull().sum()

benchmark_data.dropna(inplace=True)

# Displaying Content of each data
print(stock_data.info())
print(benchmark_data.info())

stock_data.head()
benchmark_data.head()

# Visualizing and Summarizing Facebook and Amazon's stocks
stock_data.plot(subplots=True, title = 'Stock Data')
stock_data.describe()

# Visualizing and Summarizing S&P 500
benchmark_data.plot(title = 'S&P 500')
benchmark_data.describe()

# Calculating Daily returns on stocks
stock_returns = stock_data.pct_change()

stock_returns.plot(subplots=True, title = 'Daily Returns on Stocks')
stock_returns.describe()

# S&P returns
sp_returns = pd.Series(benchmark_data['S&P 500'].pct_change())

sp_returns.plot(title = 'Daily S&P Returns')
sp_returns.describe()


# Excess Returns Calculations
excess_returns = stock_returns.sub(sp_returns, axis=0)

excess_returns.plot(subplots=True, title='Excess Returns on Stocks')
excess_returns.describe()

# Calculating the Average Excess Returns on Stocks
avg_excess_returns = excess_returns.mean()

avg_excess_returns.plot.bar(title='Mean of the Return Difference')

# Getting the Standard deviation of the Excess Returns
sd_excess_returns = excess_returns.std()

sd_excess_returns.plot.bar(title='Standard Deviation of the Return Difference')



# Applying the Sharpe Ratio 
daily_sharpe_ratio = avg_excess_returns.div(sd_excess_returns)

annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

annual_sharpe_ratio.plot.bar(title='Annualized Sharpe Ratio: Stocks vs. S&P500')