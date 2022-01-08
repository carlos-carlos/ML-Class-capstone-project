import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns

import requests
import datetime
import time
import os
import pytz
import json
import csv

from pprint import pprint

# GLOBAL SETTINGS

# Data directories
coin_dataDir = 'DATA/COINHISTDATA/'
#coin_dataDir = 'DATA/TESTDIR/' # Debug dir for testing I/O logic and/or issues. It should be a clone of the above dir.
coinMDF_dataDir = 'DATA/COMBINEDDATA/'

# Date ranges
START = 2020
END = 2022

# Helpers
idx = pd.IndexSlice

# END GLOBAL SETTINGS

# Read in the data csvs into Pandas
dir_list = os.listdir(coin_dataDir)
coins = [x.split('.')[0] for x in dir_list]

# Read in current data and set multiIndex
coin_dfs = []

for coin in coins:
    current_df = pd.read_csv(coin_dataDir + f'{coin}.csv')
    current_df.rename(columns={'Unnamed: 0': 'Dates'}, inplace=True)
    current_df['Coin'] = coin
    current_df['Dates'] = pd.to_datetime(current_df['Dates'])
    current_df.set_index(['Dates', 'Coin'], inplace=True)

    coin_dfs.append(current_df)

print("Read data for " + str(len(coin_dfs)) + " coins.")

# Create MultiIndex Dataframe for all the coins
coin_mdf = pd.concat(coin_dfs)
#print(coin_mdf.info())

# Sort index
coin_mdf.sort_index(inplace=True)
#print(coin_mdf.to_string())
#print(coin_mdf.loc[('2022-01-02', 'bitcoin')]['Close'])
#print(coin_mdf.index)


# Saves the Coin MDF to a file
dataDir = coinMDF_dataDir
isdir = os.path.isdir(dataDir)

# Save the initial pool MDF in a seprate directory for persistence and "just in case" purposes
if isdir == False:
    os.makedirs(dataDir)
    print("Directory '% s' created" % dataDir)
    coin_mdf.to_csv(f'{dataDir}CoinPool.csv')
    print(f"The initial pool of coins has been saved to {dataDir} as a MultiIndex dataframe")

else:
    coin_mdf.to_csv(f'{dataDir}CoinPool.csv')
    print(f"The initial pool of coins has been saved to {dataDir} as a MultiIndex dataframe")


# Isolate the close prices
close_df = coin_mdf.loc[idx[str(START):str(END), :], 'Close'].unstack('Coin')
#print(close_df.info)

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
#print(data.info())
#print(data.to_string())

# Drop coins with less than one year of returns
min_obs = 365
nobs = data.groupby(level='Coin').size()
keep = nobs[nobs>min_obs].index
data = data.loc[idx[keep,:], :]
#print(data.info())


# Cluster map with Seaborn
clusterMap = sns.clustermap(data.corr('spearman'), annot=True, center=0, cmap='Blues')
clusterMap.savefig('Cluster_Spearman_Blue.png')

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

# Returns for various holding periods, using the previous normalized returns, shifted back to align with current features
for t in [1,2,3,6,12]:
    data[f'target_{t}m'] = (data.groupby(level='Coin')
                            [f'return_{t}m'].shift(-t))

print(data.info())
print(data.describe())
#print(data.to_string())


# Adding time indicators
dates = data.index.get_level_values('Dates')
data['year'] = dates.year
data['month'] = dates.month

#print(dates)

