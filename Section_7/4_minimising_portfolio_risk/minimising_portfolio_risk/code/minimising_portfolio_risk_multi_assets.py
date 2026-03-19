# © Copyright, Fervent | All Rights Reserved
"""
# ========================================================
#  MINIMISING PORTFOLIO RISK - MULTIPLE ASSETS (APPLIED)
# ========================================================

# -----------
#  Beginner?
# -----------
We STRONGLY recommend using the .ipynb version instead of this .py version
The .ipynb has more explanatory notes to help and guide you through.

The .py version is largely designed for more intermediate / advanced users of
Python.
"""
# Import package dependencies
import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('../data/15stocks_price.csv')  # stock price data

# Convert dates to timestamps and set date column as the index
df['date_gsheets'] = pd.to_datetime(df['date_gsheets'])
df.set_index('date_gsheets', inplace=True)


def getPortRisk(weights):

    '''Returns the annualised standard deviation of a k asset portfolio.'''

    returns_df = df.pct_change(1).dropna()  # estimate returns for each asset

    vcv = returns_df.cov()  # being the variance covariance matrix

    var_p = np.dot(np.transpose(weights), np.dot(vcv, weights))  # variance of the multi-asset portfolio
    sd_p = np.sqrt(var_p)  # standard deviation of the multi-asset portfolio
    sd_p_annual = sd_p * np.sqrt(250)  # annualised standard deviation of the multi-asset portfolio

    return sd_p_annual


num_stocks = len(df.columns)  # being the number of stocks (this is a 'global' variable)
init_weights = [1 / num_stocks] * num_stocks  # initialise weights (x0)

# Constraint that weights in any asset j must be between 0 and 1 inclusive
bounds = tuple((0, 1) for i in range(num_stocks))

# Constraint that the sum of the weights of all assets must equate to 1
cons = ({'type' : 'eq', 'fun' : lambda x : np.sum(x) - 1})

results = minimize(fun=getPortRisk, x0=init_weights, bounds=bounds, constraints=cons)

# Check total risk of the equal weighted portfolio
getPortRisk(init_weights)

# Explore optimised weights
optimised_weights = pd.DataFrame(results['x'])
optimised_weights.index = df.columns
optimised_weights.rename(columns={optimised_weights.columns[0] : 'weights'}, inplace=True)

# Clean format of the weights so it's more readable
optimised_weights['weights_rounded'] = optimised_weights['weights'].apply(lambda x : round(x, 3))

# Notice how 7 of the 15 stocks make up 92.1% of the portfolio allocation!
print(optimised_weights['weights_rounded'].sort_values(ascending=False).cumsum())
