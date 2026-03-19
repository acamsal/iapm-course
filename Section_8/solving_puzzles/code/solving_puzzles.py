# © Copyright, Fervent | All Rights Reserved
"""
# ======================
#  SOLVING THE "PUZZLES"
# ======================

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

# Estimate the correlations across all securities
corr_matrix = returns_df.corr().round(2)

# Explore the correlations of Netflix and Tesla with all other securities
print(corr_matrix[['NFLX', 'TSLA']])

# Explore the correlations of Apple, Mastercard, and Microsoft with all other securities
print(corr_matrix[['AAPL', 'MA', 'MSFT']])

# Explore the correlations of Coca Cola and Berkshire Hathaway with all other securities
print(corr_matrix[['KO', 'BRK.B']])