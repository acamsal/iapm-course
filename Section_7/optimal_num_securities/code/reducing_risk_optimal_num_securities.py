# © Copyright, Fervent | All Rights Reserved
"""
# ========================================================
# Reducing Risk by Diversification - Number of Securities
# ========================================================

Beginner?
We STRONGLY recommend using the .ipynb version instead of this .py version
The .ipynb has more explanatory notes to help and guide you through.

The .py version is largely designed for more intermediate / advanced users of
Python.
"""

# Import package dependencies
import pandas as pd  # for data management and analysis
import numpy as np  # for numerical computations
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making charts look good
sns.set()  # implementing the seaborn plotting templates / themes

df = pd.read_csv('../data/15stocks_price.csv')  # stock price data

# Convert dates to timestamps and set date column as the index
df['date_gsheets'] = pd.to_datetime(df['date_gsheets'])
df.set_index('date_gsheets', inplace=True)

# Calculate returns for each stock
returns_df = df.pct_change(1)

# Estimating the risk of a 15 asset portfolio
num_stocks = 15
weights = [1 / num_stocks] * num_stocks  # vector (list) of weights
vcv_matrix = returns_df.cov()  # variance covariance matrix

# Calculate variance and standard deviation of the 10 asset portfolio
var_portfolio = np.dot(np.transpose(weights), np.dot(vcv_matrix, weights))
sd_portfolio = np.sqrt(var_portfolio)
sd_portfolio_annual = sd_portfolio * np.sqrt(250)

# Estimate individual stock risks for comparison
individual_risks = np.std(returns_df) * np.sqrt(250)

tickers = list(returns_df.columns)
portfolios = []

for i in range(1, len(tickers) + 1):
    portfolios.append(tickers[0:i])

# Calculate the risk of each portfolio
portfolio_risks = []
for port in portfolios:
    df = returns_df[port]

    num_stocks = len(df.columns)
    weights = [1 / num_stocks] * num_stocks
    vcv_p = df.cov()

    var_p = np.dot(np.transpose(weights), np.dot(vcv_p, weights))
    sd_p = np.sqrt(var_p)
    sd_p_annual = sd_p * np.sqrt(250)

    portfolio_risks.append(sd_p_annual)

risks_df = pd.DataFrame(portfolio_risks)
risks_df.rename(columns={risks_df.columns[0] : 'total_risk'}, inplace=True)
risks_df.reset_index(inplace=True)
risks_df.rename(columns={'index' : 'num_securities'}, inplace=True)
risks_df['num_securities'] += 1
risks_df.set_index('num_securities', inplace=True)

risks_df.plot(figsize=(12, 8), color='#39b8eb',
              title='Total Risk by Number of Securities in Portfolios')

# ============================
#   BONUS: BETTER PLOTTING
# ============================
plt.figure(figsize=(12, 8))
plt.title('Total Risk by Number of Securities in Portfolios', fontsize=16)
plt.plot(risks_df, marker='o', color='#39b8eb', markerfacecolor='#ffbd4a')
plt.xlabel('Number of Securities')
plt.ylabel('Total Portfolio Risk')
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.show()
