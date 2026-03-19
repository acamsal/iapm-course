# © Copyright, Fervent | All Rights Reserved
"""
# ========================================================
# Optimising Portfolio Weights for Target Return - Applied
# ========================================================

Beginner?
We STRONGLY recommend using the .ipynb version instead of this .py version
The .ipynb has more explanatory notes to help and guide you through.

The .py version is largely designed for more intermediate / advanced users of
Python.
"""

# Import package dependencies
import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('../data/15stocks_price.csv')
df.set_index('date_gsheets', inplace=True)

returns_df = df.pct_change(1).dropna()


# ============================
# CREATING OBJECTIVE FUNCTION
# ============================
def getPortReturn(weights):
    """
    Returns the Annualised Expected Return of a portfolio.
    Annualises the return using the 'crude' method.
    """
    exp_ret_portfolio = np.dot(np.transpose(weights), returns_df.mean()) * 250

    return exp_ret_portfolio


# Create a vector of equal weights as the initial (init) guess
num_stocks = len(returns_df.columns)
init_weights = [1 / num_stocks] * num_stocks

# Calculate the expected return on the portfolio of equal weights
getPortReturn(init_weights)

# Set a target return (note: the lecture starts with a target_return = 0.3)
target_return = 0.4

# ============================
#   SETTING UP CONSTRAINTS
# ============================

# Create the constraint that the weight of any asset i must be between 0 and 1 inclusive.
bounds = tuple((0, 1) for i in range(num_stocks))

# Setup the other 2 constraints
cons = (
    # Sum of weights must equate to 1
    {'type' : 'eq', 'fun' : lambda w : np.sum(w) - 1},

    # Difference between expected return and target must be equal to 0.
    {'type' : 'eq', 'fun' : lambda x : x.dot(returns_df.mean()) * 250 - target_return})

results = minimize(fun=getPortReturn,  # being the objective function
                   x0=init_weights,  # being the initial guess
                   # bounds: being the constraint that the weight of any asset i
                   #         must be between 0 and 1 inclusive
                   bounds=bounds,
                   constraints=cons)  # being the other 2 constraints (see 'cons' in Line 57)

# Test the output by passing in the optimised weights into getPortReturn()
getPortReturn(weights=results['x'])

# Store the optimised weights as a dataframe object
optimised_weights = pd.DataFrame(results['x'])

# Set the tickers as the index of `optimised_weights` by using the column names of `returns_df`.
optimised_weights.index = returns_df.columns

print(optimised_weights)
