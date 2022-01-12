import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.ticker import FuncFormatter

import datetime
import time
import os
import pytz
import json
import csv

from pprint import pprint

# GLOBAL SETTINGS

# Data directories
#coin_dataDir = 'DATA/TESTDIR/' # Debug dir for testing I/O logic and/or issues. It should be a clone of the above dir.
coin_dataDir = 'DATA/COMBINEDDATA/'
plot_dataDir = 'DATA/INITIAL_INSIGHTS/'

# Date ranges
START = 2020
END = 2022

# Helpers
idx = pd.IndexSlice
sns.set_style('whitegrid')
np.random.seed(42)

# END GLOBAL SETTINGS

# Read in MDF with initial coin pool
cpool_mdf = pd.read_csv(coin_dataDir + 'CoinPool.csv')
cpool_mdf.rename(columns={'Unnamed: 0': 'Dates'}, inplace=True)
cpool_mdf['Dates'] = pd.to_datetime(cpool_mdf['Dates'])
cpool_mdf.set_index(['Dates', 'Coin'], inplace=True)

print('Initial Base Data:')
print(cpool_mdf.info())


# Isolate the close prices and calculate daily returns
returns_df = cpool_mdf.loc[idx[str(START):str(END), :], 'Close'].unstack('Coin').pct_change()
print(returns_df.info())
#print(returns_df.to_string())

# Winsorize at 2.5% and 97.5% quantiles
returns_df = returns_df.clip(lower=returns_df.quantile(q=.025),
                       upper=returns_df.quantile(q=.975),
                       axis=1)

print(returns_df.info())

# Drop coins and adates that do not have complete data for 95%  of the time period
returns_df = returns_df.dropna(thresh=int(returns_df.shape[0] * .95), axis=1)
returns_df = returns_df.dropna(thresh=int(returns_df.shape[1] * .95))
print(returns_df.info())

# impute any remaining missing values using daily average returns
daily_avg = returns_df.mean(1)
returns_df = returns_df.apply(lambda x: x.fillna(daily_avg))
print(returns_df.info())

#for c in returns_df:
#    print(c + " " + str(returns_df[c].isnull().sum()))

# Use the defaults for PCA to computer principle components of the returns
pca = PCA(n_components='mle')
fitted_returns = pca.fit(returns_df)

print(fitted_returns)

# Plot the Explained Variance and Top Factors
# The top Factors found by PCA can be used as the "Risk Factors" for our model instead of the FAMA-French Risk Factors
fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
title = 'Explained Variance Ratio by Top Factors'
var_expl = pd.Series(pca.explained_variance_ratio_)
var_expl.index += 1
var_expl.iloc[:15].sort_values().plot.barh(title=title,
                                           ax=axes[0])
var_expl.cumsum().plot(ylim=(0, 1),
                       ax=axes[1],
                       title='Cumulative Explained Variance',
                       xlim=(1, 300))
axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
sns.despine()
fig.tight_layout()
fig.savefig(plot_dataDir + 'PCA_Exp_Var_&_Cumulative_Exp_Var.png')

'''
It appears the most important factor explains well over 40% of the daily return variation.
Furthermore, it looks like about 10 factors explain 80% of the returns in our cross section of 40 crypto coins.
About 5 coins explain 60% of the returns. 
Though the initial pool of coins was larger than 40 we dropped several due to lack of data.
'''

# Isolate the first 2 factors
risk_factors = pd.DataFrame(pca.transform(returns_df)[:, :2],
                            columns=['Principal Component 1', 'Principal Component 2'],
                            index=returns_df.index)
print(risk_factors.info())

# Make sure the first 2 factors are really uncorrelated
factor_corr_1_2 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 2'])
print(factor_corr_1_2)


with sns.axes_style('white'):
    risk_factors.plot(subplots=True,
                      figsize=(14, 8),
                      title=risk_factors.columns.tolist(),
                      legend=False,
                      rot=0,
                      lw=1,
                      xlim=(risk_factors.index.min(),
                            risk_factors.index.max()))

    sns.despine()
    plt.tight_layout()
    plt.savefig(plot_dataDir + 'Principle_component_volatillity.png')
