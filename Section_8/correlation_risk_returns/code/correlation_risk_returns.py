# © Copyright, Fervent | All Rights Reserved
"""
# ===============================
#  CORRELATION, RISK, AND RETURNS
# ===============================

# -----------
#  Beginner?
# -----------
We STRONGLY recommend using the .ipynb version instead of this .py version
The .ipynb has more explanatory notes to help and guide you through.

The .py version is largely designed for more intermediate / advanced users of
Python.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Stocks we'll work with:
# Arlington Plc (ALTN)
# Brighton Inc (BTON)

def get_port_weights():
    """Returns the weights to be invested in securities belonging to a 2 asset portfolio.
    Each ensuing weight is decremented (incremented) by 0.1 for ALTN (BTON)."""
    
    # Weights of Arlington == 1 through to 0 in decrements of 0.1
    weights_altn = list(np.round(np.arange(1, -0.1, -0.1), 2))
    
    # Weights of Brighton == 0 through to 1 in increments of 0.1
    weights_bton = weights_altn[:]
    weights_bton.reverse()
    
    return weights_altn, weights_bton


# Explore the weights of Arlington and Brighton in different portfolios
get_port_weights()


def get_port_returns_risk(expected_returns, total_risks, correlation, weights):
    """Returns the portfolio returns and risks of multiple portfolio iterations.

    Args:
        expected_returns ([type: list of length 2]): [expected returns of security a and b]
        total_risks ([type: list of length 2]): [total risk of security a and b]
        correlation ([type: float]): [correlation between securities a and b]
        weights ([type: tuple of 2 lists]): [weights in security a and b for each portfolio]

    Returns:
        [type: tuple of 2 lists]: [portfolio returns and portfolio risks of different portfolio iterations]
    """
    
    port_returns = []
    port_risks = []
    for i in range(len(weights[0])):
        # Asset Weights
        w_altn = weights[0][i]
        w_bton = weights[1][i]
        
        # Asset Expected returns
        exp_r_altn = expected_returns[0]
        exp_r_bton = expected_returns[1]
        
        # Asset Risks
        risk_altn = total_risks[0]
        risk_bton = total_risks[1]
        
        # Portfolio Return
        port_ret = w_altn * exp_r_altn + w_bton * exp_r_bton
        
        # Portfolio Risk
        port_risk = np.sqrt((w_altn ** 2) * (risk_altn ** 2) \
                                + (w_bton ** 2) * (risk_bton ** 2) \
                                    + (2 * w_altn * w_bton * risk_altn * risk_bton * correlation))

        port_returns.append(port_ret)
        port_risks.append(port_risk)
    
    return port_returns, port_risks


def plot_my_scatter(expected_returns, total_risks, weights, correlation):

    port_returns, port_risks = get_port_returns_risk(expected_returns, total_risks, correlation, weights)
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Portfolio Return vs. Portfolio Risk when ρ = {correlation}", fontsize=18)
    plt.scatter(port_risks, port_returns, color='#ffbd4a', zorder=2)
    plt.plot(port_risks, port_returns, color='#39b8eb', zorder=1, linewidth=2)
    plt.xlabel("Portfolio Risk (σ)")
    plt.ylabel("Expected Portfolio Return (E[r])")
    plt.ion()
    
    return


# Stock parameters
expected_r_arlington = .12
expected_r_brighton = .18
risk_arlington = .2 
risk_brighton = .3

plot_my_scatter(expected_returns=[expected_r_arlington,
                                  expected_r_brighton],
               total_risks=[risk_arlington,
                           risk_brighton],
               weights=get_port_weights(),
               correlation=.3)