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

# Lag periods for monthly or daily close prices to make returns for various periods

# Monthly close lags
#lags = {'lagged_returns':[1, 2, 3, 6, 9, 12],
#        "recent_month_returns":[2, 3, 6, 9, 12],
#        "target_forward_returns": [1,2,3,6,12]}

# Daily close lags
#lags = {'lagged_returns':[1, 7, 14, 30, 61, 91,183,274,365 ],
#        "recent_month_returns":[61, 91, 183, 274, 365],
#        "target_forward_returns": [30,61,91,183,365]}

# Helpers
idx = pd.IndexSlice

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
