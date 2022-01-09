import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt


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


# Isolate the close prices
close_df = cpool_mdf.loc[idx[str(START):str(END), :], 'Close'].unstack('Coin')

# Optionally resamble the daily closes to monthly instead of daily data
close_df = close_df.resample('M').last()
#print(close_df.to_string())

# Calculate lagged returns
outlier_threshold = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]

# This block stacks the wide MDF to long formant while also:
# Winsorizing outliers in the returns at the 1% and 99% levels
# Capping outliers and the aforementioned levels
# Normalize the returns via geometric mean
for lag in lags:
    data[f'return_{lag}m'] = (close_df
                           .pct_change(lag)
                           .stack()
                           .pipe(lambda x: x.clip(lower=x.quantile(outlier_threshold),
                                                  upper=x.quantile(1-outlier_threshold)))
                           .add(1)
                           .pow(1/lag)
                           .sub(1)
                           )

# Resulting in compunded daily returns for the six monthly periods in the lags list above
data = data.swaplevel().dropna()

# Drop coins with less than one year of returns
min_obs = 12 # number of months
nobs = data.groupby(level='Coin').size()
keep = nobs[nobs>min_obs].index
data = data.loc[idx[keep,:], :]
print(data.to_string())



# Cluster map with Seaborn
clusterMap = sns.clustermap(data.corr('spearman'), annot=True, center=0, cmap='Blues')
clusterMap.savefig(plot_dataDir + 'Cluster_Spearman_Blue.png')

print('Coins with Unique Values:')
print(data.index.get_level_values('Coin').nunique())

# Compute momentum factor based on difference between 3 and 12 month returns
# And for differences between all periods and the most recent month returns

# Most recent month and the rest
for lag in [2,3,6,9,12]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)

# 3 and 12 month returns
data["momentum_3_12"] = data["return_12m"].sub(data.return_3m)

# Move historical returns up to current period so they can be used as features
for t in range(1,7):
    data[f'return_1m_t-{t}'] = data.groupby(level='Coin').return_1m.shift(t)

# Target forward returns for various holding periods, using the previous normalized returns
for t in [1,2,3,6,12]:
    data[f'target_{t}m'] = (data.groupby(level='Coin')
                            [f'return_{t}m'].shift(-t))

print(data.info())
print(data.describe())

# Check return distributions
dist_mdf = data
#dist_mdf.drop('bitcoin', level=1, axis=0, inplace=True)
#dist_mdf.drop('ethereum', level=1, axis=0, inplace=True)
distPlot = sns.distplot(dist_mdf.return_1m)
distPlot = sns.despine()
#plt.tight_layout()
distPlot = plt.savefig(plot_dataDir + 'Price_Distribution_Plot.png')

# Adding time indicators
dates = data.index.get_level_values('Dates')
data['year'] = dates.year
data['month'] = dates.month
#print(dates)

