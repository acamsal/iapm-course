# © Copyright, Fervent | All Rights Reserved
"""
# =====================================================
# ESTIMATING PORTFOLIO RISK (MULTIPLE ASSETS) - APPLIED
# =====================================================

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

df = pd.read_csv('../data/10stocks_price.csv')

# Convert string format dates to pandas datetime objects
# This helps us explore the time series dataset (e.g., determining start and end of the sample)
df['date_gsheets'] = pd.to_datetime(df['date_gsheets'])

df['date_gsheets'].describe()

# Exploring the data
df.describe()

# Exploring the datatypes, checking for NaNs
df.info()

# Set the date column as the index to apply formulas to the dataframe as a whole
df.set_index('date_gsheets', inplace=True)

# Calculate returns for each stock, at each time
returns_df = df.pct_change(1)

# Create a vector of equal weights (\omega_1 = \omega_1 = ... = \omega_10)
num_stocks = 10
weights = [1 / num_stocks] * num_stocks

# Calculate the variance covariance matrix
vcv_matrix = returns_df.cov()

# Calculate the variance of the 10 asset portfoio
var_p = np.dot(np.transpose(weights), np.dot(vcv_matrix, weights))

# Calculate the total risk (standard deviation) of the 10 asset portfolio
sd_p = np.sqrt(var_p)

# Calculate the Annualised Standard Deviation of the 10 asset portfolio
sd_p_annual = sd_p * np.sqrt(250)

# Compare the Portfolio Risk with the individual risks of each security
individual_risks = np.std(returns_df) * np.sqrt(250)
