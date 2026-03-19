# © Copyright, Fervent | All Rights Reserved
"""
# =================================
#  ESTIMATING CORRELATION - APPLIED
# =================================

# -----------
#  Beginner?
# -----------
We STRONGLY recommend using the .ipynb version instead of this .py version
The .ipynb has more explanatory notes to help and guide you through.

The .py version is largely designed for more intermediate / advanced users of
Python.
"""
# Import packages
import pandas as pd
import numpy as np

df = pd.read_csv('../data/15stocks_price.csv')  # stock price data

# Convert dates to timestamps and set date column as the index
df['date_gsheets'] = pd.to_datetime(df['date_gsheets'])
df.set_index('date_gsheets', inplace=True)

# Calculate returns for all securities
returns_df = df.pct_change(1)

# Drop / delete missing observations
returns_df.dropna(inplace=True)

# Estimate the covariance between Apple and Coca Cola
cov_aapl_ko = np.cov(returns_df['AAPL'], returns_df['KO'])[0][1]

# Estimate the standard deviation of Apple and Coca Cola
std_aapl = returns_df['AAPL'].std()
std_ko = returns_df['KO'].std()

# Estimate the correlation between Apple and Coca Cola
corr_aapl_ko = cov_aapl_ko / (std_aapl * std_ko)

# Estimate the correlations across all securities
corr_matrix = returns_df.corr().round(2)