import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt
import talib
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

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
model_dataDir = 'DATA/MODELDATA/'


isdir = os.path.isdir(model_dataDir)

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
print("After dropping coins with less than 2 years of data")
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


#Plus/Minus Directional Index
def compute_diplus(coin_data):
    real = talib.PLUS_DI(coin_data.High,
                     coin_data.Low,
                     coin_data.Close,
                     timeperiod=14)

    return real

def compute_diminus(coin_data):
    real = talib.MINUS_DI(coin_data.High,
                     coin_data.Low,
                     coin_data.Close,
                     timeperiod=14)

    return real

prices_mdf["DI_PLUS"] = (prices_mdf.groupby('Coin', group_keys=False).apply(compute_diplus))
prices_mdf["DI_MINUS"] = (prices_mdf.groupby('Coin', group_keys=False).apply(compute_diminus))

# Compute lagged returns and Winsorize
lags = [1, 7, 14, 30, 60, 90]
q = 0.0001

for lag in lags:
    prices_mdf[f'return_{lag}d'] = (prices_mdf.groupby(level='Coin').Close
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(q),
                                                       upper=x.quantile(1 - q)))
                                .add(1)
                                .pow(1 / lag)
                                .sub(1)
                                )
# Shift lagged returns
for t in [1, 2, 3, 4, 5]:
    for lag in [1, 7, 14, 30, 60, 90]:
        prices_mdf[f'return_{lag}d_lag{t}'] = (prices_mdf.groupby(level='Coin')
                                           [f'return_{lag}d'].shift(t * lag))

# Generate target forward returns
for t in [1, 7, 14, 30, 60, 90]:
    prices_mdf[f'target_{t}d'] = prices_mdf.groupby(level='Coin')[f'return_{t}d'].shift(-t)

# Create dummy time variables
prices_mdf['year'] = prices_mdf.index.get_level_values('Dates').year
prices_mdf['month'] = prices_mdf.index.get_level_values('Dates').month

prices_mdf = pd.get_dummies(prices_mdf,
                        columns=['year', 'month'],
                        prefix=['year', 'month'],
                        prefix_sep=['_', '_'],
                        drop_first=True)

print(prices_mdf.info(null_counts=True))

# Save the model data
if isdir == False:
    os.makedirs(model_dataDir)
    print("Directory '% s' created" % model_dataDir)
    prices_mdf.to_csv(f'{model_dataDir}ModelData.csv')
    print(f"The model data has been saved to {model_dataDir} as a MultiIndex dataframe")

else:
    prices_mdf.to_csv(f'{model_dataDir}ModelData.csv')
    print(f"The model data has been saved to {model_dataDir} as a MultiIndex dataframe")

# DATASET INSIGHTS AND VISUALIZATION PLOTS

# Correlation Cluster map of the Returns
returns = prices_mdf.loc[:, 'return_1d':"return_90d"]
#clusterMap = sns.clustermap(returns.corr('spearman'), annot=True, center=0, cmap='Blues')
#clusterMap.savefig(plot_dataDir + 'Cluster_Spearman_Blue.png')

print('Coins with Unique Values:')
print(returns.index.get_level_values('Coin').nunique())

# Check return distributions
for x in returns.loc[:,:'return_90d'].columns:
    sns_distPlot = sns.distplot(returns[f'{x}'])
    fig = sns_distPlot.get_figure()
    sns.despine()
    fig.savefig(plot_dataDir + f'{x}Distplot.png')


# Spearman Ranks and scatter plots for factors
target = 'target_7d'
price_copy = prices_mdf.copy()

# Relative Strength Index (RSI)
print("RSI FEATURE INFO")
price_copy.loc[:, 'rsi_signal'] = pd.cut(price_copy.RSI, bins=[0, 30, 70, 100])
print("RSI Distributions")
print(price_copy.groupby('rsi_signal')['target_7d'].describe().to_string())

metric = "RSI"
df = price_copy[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print("RSI Spearman")
print(f'{r:,.2%} ({p:.2%})')
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'RSI_Scatter_Plot.png')

# Average Directional Moving Index (ADX)
print("ADX FEATURE INFO")
price_copy.loc[:, 'adx_signal'] = pd.cut(price_copy.ADX, bins=[0, 25, 50, 75, 100])
print("ADX Distributions")
print(price_copy.groupby('adx_signal')['target_7d'].describe().to_string())

metric = "ADX"
df = price_copy[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print("ADX Spearman")
print(f'{r:,.2%} ({p:.2%})')
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'ADX_Scatter_Plot.png')

# Directional Indices (DM+/-)
metric = "DI_PLUS"
df = price_copy[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print("DI Plus Spearman")
print(f'{r:,.2%} ({p:.2%})')
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'DI_PLUS_Scatter_Plot.png')

metric = "DI_MINUS"
df = price_copy[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print("DI Minus Spearman")
print(f'{r:,.2%} ({p:.2%})')
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'DI_MINUS_Scatter_Plot.png')


# Bollinger Bands
metric = 'BB_low'
df = price_copy[[metric, target]].dropna()
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'BB_Low_Scatter_Plot.png')
r, p = spearmanr(df[metric], df[target])
print("Lower BB Spearman")
print(f'{r:,.2%} ({p:.2%})')

metric = 'BB_high'
df = price_copy[[metric, target]].dropna()
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'BB_High_Scatter_Plot.png')
r, p = spearmanr(df[metric], df[target])
print("Upper BB Spearman")
print(f'{r:,.2%} ({p:.2%})')

# Active True Range (ATR)
metric = 'ATR'
j=sns.jointplot(x=metric, y=target, data=price_copy)
j.savefig(plot_dataDir + 'ATR_Scatter_Plot.png')
df = price_copy[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print("ATR Spearman")
print(f'{r:,.2%} ({p:.2%})')

# Moving Average Convegeance Divergeance (MACD)
metric = 'MACD'
df = price_copy[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print("MACD Spearman")
print(f'{r:,.2%} ({p:.2%})')
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'MACD_Scatter_Plot.png')

# Stochastic Oscillator
metric = 'STOCH'
df = price_copy[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print("STOCH Spearman")
print(f'{r:,.2%} ({p:.2%})')
j=sns.jointplot(x=df[metric], y=df[target], data=df)
j.savefig(plot_dataDir + 'STOCH_Scatter_Plot.png')


'''
# Distribution plots and statistics
print("RETURNS PERCENTILES")
returns = prices_mdf.groupby(level='Coin').Close.pct_change()
percentiles=[.0001, .001, .01]
percentiles+= [1-p for p in percentiles]
print(returns.describe(percentiles=percentiles).iloc[2:].to_frame('percentiles'))

# RSI distplot
#RSI_ax = sns.distplot(prices_mdf.RSI.dropna())
#RSI_ax.axvline(30, ls='--', lw=1, c='k')
#RSI_ax.axvline(70, ls='--', lw=1, c='k')
#RSI_ax.set_title('RSI Distribution with Signal Threshold')
#plt.tight_layout()
#plt.savefig(plot_dataDir + 'RSI Distribution with Signal Threshold.png')

# Bollinger Bands distplot
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
print("MACD Percentiles")
print(prices_mdf
      .MACD
      .describe(percentiles=[.001, .01, .02, .03, .04, .05, .95, .96, .97, .98, .99, .999])
      .apply(lambda x: f'{x:,.1f}'))

#MACD_dist = sns.distplot(prices_mdf.MACD.dropna())
#plt.savefig(plot_dataDir + 'MACD_Distribution.png')

# ADX distribution plot
#ADX_dist = sns.distplot(prices_mdf.ADX.dropna())
#ADX_dist.axvline(25, ls='--', lw=1, c='k')
#ADX_dist.axvline(50, ls='--', lw=1, c='k')
#ADX_dist.axvline(75, ls='--', lw=1, c='k')

#ADX_dist.set_title('ADX Distribution with Signal Threshold')
#plt.savefig(plot_dataDir + 'ADX_Distribution.png')

# Stochastic Oscillator distplot
print("STOCH percentiles")
print(prices_mdf
      .STOCH
      .describe(percentiles=[.001, .01, .02, .03, .04, .05, .95, .96, .97, .98, .99, .999])
      .apply(lambda x: f'{x:,.1f}'))



#STOCH_ax = sns.distplot(prices_mdf.STOCH.dropna())
#plt.savefig(plot_dataDir + 'STOCH Distribution with Signal Threshold.png')

# Directional Indicators
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
sns.distplot(prices_mdf.DI_PLUS.dropna(), ax=axes[0])
sns.distplot(prices_mdf.DI_MINUS.dropna(), ax=axes[1])
plt.tight_layout()
plt.savefig(plot_dataDir + 'Directional Indicators.png')
'''


'''
# Compute Exponential Moving Averages (EMA)
ema_periods = [9, 20, 50, 100, 200]

for p in ema_periods:

    def compute_ema(coin_data):
        real = talib.EMA(coin_data.Close, timeperiod=p)
        return real

    prices_mdf[f'EMA{p}'] = (prices_mdf
                           .groupby(level='Coin',group_keys=False)
                           .apply(compute_ema))
'''