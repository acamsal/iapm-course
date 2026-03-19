# © Copyright, Fervent | All Rights Reserved
"""
# ===============================================================
# Reducing Risk by Diversification - Berkshire Hathaway Portfolio
# ===============================================================
"""
import pandas as pd

brk_portfolio = pd.read_csv('../data/brk_portfolio.csv')
sorted_by_weights = brk_portfolio.sort_values(by='weights', ascending=False)
sorted_by_weights['cumulative_weights'] = sorted_by_weights['weights'].cumsum()
print(sorted_by_weights)
