import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt
import talib
from sklearn.feature_selection import mutual_info_regression



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

MONTH = 30
YEAR = 12 * MONTH

# Helpers
idx = pd.IndexSlice
sns.set_style('whitegrid')

# END GLOBAL SETTINGS

# Read in MDF with initial coin pool
cpool_mdf = pd.read_csv(coin_dataDir + 'CoinPool.csv')
cpool_mdf.rename(columns={'Unnamed: 0': 'Dates'}, inplace=True)
cpool_mdf['Dates'] = pd.to_datetime(cpool_mdf['Dates'])
cpool_mdf.set_index(['Dates', 'Coin'], inplace=True)

print('Initial Base Data:')
print(cpool_mdf.info())


ohlcv = ['Open','High','Low','Close','Volume']
prices_mdf = (cpool_mdf
              .loc[idx[str(START):str(END), :], ohlcv]
              .swaplevel()
              .sort_index())



print(prices_mdf.info())

# want at least 2 years of data
min_obs = 2 * YEAR

# have this much per ticker
nobs = prices_mdf.groupby(level='Coin').size()

# keep those that exceed the limit
keep = nobs[nobs > min_obs].index

prices_mdf = prices_mdf.loc[idx[keep, :], :]
print(prices_mdf.info())

# These lines can be used to select coins based on volume from a greater pool if taken without the initial
# criteria I selected for the coins for this project
# They limit the trading universe of coins based on volume for the last month
#prices_mdf['Volume'] = prices_mdf[['Close', 'Volume']].prod(axis=1)
#prices_mdf['Volume_1m'] = (prices.Volume.groupby('Coin')
#                           .rolling(window=30, level='Dates')
#                           .mean()).values

# Compute Technical Analysis Indicators to be use as momentum alpha factors

# Relative Strength Index (RSI)
prices_mdf['RSI'] = prices_mdf.groupby(level='Coin').Close.apply(talib.RSI)

# Bollinger Bands
def compute_bb(close):
    high, mid, low = talib.BBANDS(close, timeperiod=20)
    return pd.DataFrame({'BB_high': high, 'BB_mid':mid, 'BB_low': low}, index=close.index)

prices_mdf = (prices_mdf.join(prices_mdf
                      .groupby(level='Coin')
                      .Close
                      .apply(compute_bb)))

# Average True Range (ATR)
def compute_atr(coin_data):
    df = talib.ATR(coin_data.High, coin_data.Low,
             coin_data.Close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())

prices_mdf['ATR'] = (prices_mdf.groupby('Coin', group_keys=False)
                 .apply(compute_atr))

# Moving Average Convergence Divergence (MACD)
def compute_macd(close):
    macd = talib.MACD(close)[0]
    return (macd - np.mean(macd))/np.std(macd)

prices_mdf['MACD'] = (prices_mdf
                  .groupby('Coin', group_keys=False)
                  .Close
                  .apply(compute_macd))


# Stochastic Oscillator (STOCH)
def compute_stoch(coin_data):
    slowk, slowd = talib.STOCH(coin_data.High,
                         coin_data.Low,
                         coin_data.Close,
                         fastk_period=14,
                         slowk_period=3,
                         slowk_matype=0,
                         slowd_period=3,
                         slowd_matype=0)

    return slowd/slowk

prices_mdf['STOCH'] = (prices_mdf
                       .groupby('Coin', group_keys=False)
                       .apply(compute_stoch))

# Average Directional Index (ADX)
def compute_adx(coin_data):
    real = talib.ADX(coin_data.High,
                     coin_data.Low,
                     coin_data.Close,
                     timeperiod=14)

    return real

prices_mdf['ADX'] = (prices_mdf
                     .groupby('Coin', group_keys=False)
                     .apply(compute_adx))

print(prices_mdf.info())

# Compute Exponential Moving Averages (EMA)
ema_periods = [9, 20, 50, 100, 200]

for p in ema_periods:

    def compute_ema(coin_data):
        real = talib.EMA(coin_data.Close, timeperiod=p)
        return real

    prices_mdf[f'EMA{p}'] = (prices_mdf
                           .groupby(level='Coin',group_keys=False)
                           .apply(compute_ema))


print(prices_mdf.info())

# Visualize the distribution in the features

# RSI distplot
#RSI_ax = sns.distplot(prices_mdf.RSI.dropna())
#RSI_ax.axvline(30, ls='--', lw=1, c='k')
#RSI_ax.axvline(70, ls='--', lw=1, c='k')
#RSI_ax.set_title('RSI Distribution with Signal Threshold')
#plt.tight_layout()
#plt.savefig(plot_dataDir + 'RSI Distribution with Signal Threshold.png')

# Bollinger Bands displot
prices_mdf['BB_high'] = prices_mdf.BB_high.sub(prices_mdf.Close).div(prices_mdf.BB_high).apply(np.log1p)
prices_mdf['BB_low'] = prices_mdf.Close.sub(prices_mdf.BB_low).div(prices_mdf.Close).apply(np.log1p)

#fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
#sns.distplot(prices_mdf.BB_low.dropna(), ax=axes[0])
#sns.distplot(prices_mdf.BB_high.dropna(), ax=axes[1])
#plt.tight_layout()

#plt.savefig(plot_dataDir + 'Bollinger_Band_Distribution.png')

# Average True Range
#atr_plot = sns.distplot(prices_mdf.ATR.dropna())
#fig = atr_plot.get_figure()
#fig.savefig(plot_dataDir + 'ATR_Distribution.png')

# MACD distribution
print(prices_mdf
      .MACD
      .describe(percentiles=[.001, .01, .02, .03, .04, .05, .95, .96, .97, .98, .99, .999])
      .apply(lambda x: f'{x:,.1f}'))

#MACD_dist = sns.distplot(prices_mdf.MACD.dropna())
#plt.savefig(plot_dataDir + 'MACD_Distribution.png')

ADX_dist = sns.distplot(prices_mdf.ADX.dropna())
ADX_dist.axvline(25, ls='--', lw=1, c='k')
ADX_dist.axvline(50, ls='--', lw=1, c='k')
ADX_dist.axvline(75, ls='--', lw=1, c='k')

ADX_dist.set_title('ADX Distribution with Signal Threshold')
plt.savefig(plot_dataDir + 'ADX_Distribution.png')


'''
def compute_ema(coin_data):
    real = talib.EMA(coin_data.Close, timeperiod=20)
    return real

prices_mdf['EMA20'] = (prices_mdf
                       .groupby(level='Coin',group_keys=False)
                       .apply(compute_ema))
'''


'''
# Isolate the close prices
close_df = cpool_mdf.loc[idx[str(START):str(END), :], 'Close'].unstack('Coin')

# Optionally resamble the daily closes to monthly instead of daily data
close_df = close_df.resample('M').last()
#print(close_df.to_string())

# Calculate lagged returns
outlier_threshold = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6]

# This block stacks the wide MDF to long formant while also:
# Winsorizing outliers in the returns at the 1% and 99% levels
# Capping outliers and the aforementioned levels
# Normalize the returns via geometric mean
# calculate lagged monthly returns for the periods above
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
#print(data.to_string())



# Cluster map with Seaborn
#clusterMap = sns.clustermap(data.corr('spearman'), annot=True, center=0, cmap='Blues')
#clusterMap.savefig(plot_dataDir + 'Cluster_Spearman_Blue.png')

print('Coins with Unique Values:')
print(data.index.get_level_values('Coin').nunique())

# Compute momentum factor based on difference between 3 and 12 month returns
# And for differences between all periods and the most recent month returns

# Most recent month and the rest
for lag in [2,3,6]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)

# Returns Momentum Factors
data["momentum_1_6"] = data["return_6m"].sub(data.return_1m) # 6 minus 1 months
data["momentum_1_3"] = data["return_3m"].sub(data.return_1m) # 3 minus 1 months
data["momentum_3_6"] = data["return_6m"].sub(data.return_3m) # 6 minus 3 months

# Move historical returns up to current period so they can be used as features
for t in range(1,5):
    data[f'return_1m_t-{t}'] = data.groupby(level='Coin').return_1m.shift(t)

#print(data.to_string())
# Target forward returns for various holding periods, using the previous normalized returns
for t in [1,2,3,6]:
    data[f'target_{t}m'] = (data.groupby(level='Coin')
                            [f'return_{t}m'].shift(-t))

print(data.info())
print(data.describe())
print(data.to_string())


# Check return distributions
for x in data.loc[:,:'return_6m'].columns:
    print(x)
    sns_distPlot = sns.distplot(data[f'{x}'])
    sns.despine()
    plt.savefig(plot_dataDir + f'{x}Distplot.png')



# Adding time indicators
dates = data.index.get_level_values('Dates')
data['year'] = dates.year
data['month'] = dates.month
#print(dates)

print(data.iloc[0:369].to_string())
print(data.iloc[-369:-1].to_string())


'''
