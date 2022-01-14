import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
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
print(cpool_mdf.columns)


# Isolate the close prices and calculate daily returns
returns_df = cpool_mdf.loc[idx[str(START):str(END), :], 'Close'].unstack('Coin').pct_change()

# Winsorize at 2.5% and 97.5% quantiles
returns_df = returns_df.clip(lower=returns_df.quantile(q=.025),
                       upper=returns_df.quantile(q=.975),
                       axis=1)

# Drop coins and adates that do not have complete data for 95%  of the time period
returns_df = returns_df.dropna(thresh=int(returns_df.shape[0] * .95), axis=1)
returns_df = returns_df.dropna(thresh=int(returns_df.shape[1] * .95))

# Saved the returns as they are for comparison to Eigenportfolios at the end of the script
base_returns = returns_df
#print("After Dropping coins for missing data")
#print(returns_df.info())

# impute any remaining missing values using daily average returns
daily_avg = returns_df.mean(1)
returns_df = returns_df.apply(lambda x: x.fillna(daily_avg))
#print(returns_df.info())

# Verify no NaNs left in the data
#for c in returns_df:
#    print(c + " " + str(returns_df[c].isnull().sum()))

# Use the defaults for PCA to computer principle components of the returns
pca = PCA(n_components='mle')
fitted_returns = pca.fit(returns_df)
#print(fitted_returns)

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

# Isolate the first X factors
risk_factors = pd.DataFrame(pca.transform(returns_df)[:, :7],
                            columns=['Principal Component 1', 'Principal Component 2','Principal Component 3',
                                     'Principal Component 4', 'Principal Component 5','Principal Component 6',
                                     'Principal Component 7'],
                            index=returns_df.index)
#print(risk_factors.info())



# Verify factor correlation or non-correlation
factor_corr_1_1 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 1'])
factor_corr_1_2 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 2'])
factor_corr_1_3 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 3'])
factor_corr_1_4 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 4'])
factor_corr_1_5 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 5'])
factor_corr_1_6 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 6'])
factor_corr_1_7 = risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 7'])

factor_corr_2_3 = risk_factors['Principal Component 2'].corr(risk_factors['Principal Component 3'])
factor_corr_2_4 = risk_factors['Principal Component 2'].corr(risk_factors['Principal Component 4'])
factor_corr_2_5 = risk_factors['Principal Component 2'].corr(risk_factors['Principal Component 5'])




print("Factor Correlations")
print("Factors 1 & 1 " + str(factor_corr_1_1))
print("Factors 1 & 2 " + str(factor_corr_1_2))
print("Factors 1 & 3 " + str(factor_corr_1_3))
print("Factors 2 & 3 " + str(factor_corr_2_3))
print("Factors 1 & 4 " + str(factor_corr_1_4))
print("Factors 1 & 5 " + str(factor_corr_1_5))
print("Factors 1 & 6 " + str(factor_corr_1_6))
print("Factors 1 & 7 " + str(factor_corr_1_7))

print("Factors 2 & 4 " + str(factor_corr_2_4))
print("Factors 2 & 5 " + str(factor_corr_2_5))

'''
In this case factors 1 and 2 are pretty correlated. But factors 1 and 3 are far less correlated, factors 2 and 3 are
less correlated than factors 1 and 2.
'''

#  Plot with Seaborn
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


# Same as above but with 100 trials

# Isolate the close prices and calculate daily returns
returns_df = cpool_mdf.loc[idx[str(START):str(END), :], 'Close'].unstack('Coin').pct_change()
#print("Regenerate daily Returns")
#print(returns_df.info())

# Winsorize at 2.5% and 97.5% quantiles
returns_df = returns_df.clip(lower=returns_df.quantile(q=.025),
                       upper=returns_df.quantile(q=.975),
                       axis=1)
#print("Winsorize again daily Returns")
#print(returns_df.info())

# Fit PCA and do the trials
pca = PCA()
n_trials, n_samples = 100, 500
explained = np.empty(shape=(n_trials, n_samples))

# The trials
for trial in range(n_trials):
    returns_sample = returns_df.sample(n=n_samples)
    returns_sample = returns_sample.dropna(thresh=int(returns_sample.shape[0] * .95), axis=1)
    returns_sample = returns_sample.dropna(thresh=int(returns_sample.shape[1] * .95))
    daily_avg = returns_sample.mean(1)
    returns_sample = returns_sample.apply(lambda x: x.fillna(daily_avg))
    #print("After Dropping coins for missing data")
    #print(returns_sample.info())
    fitted_returns = pca.fit(returns_sample)
    explained[trial, :len(pca.components_)] = fitted_returns.explained_variance_ratio_

#pprint(explained)

explained = pd.DataFrame(explained, columns=list(range(1, explained.shape[1] + 1)))
#print("All the explained covariance PCA components")
#print(explained.info())

# Plot with Seaborn
fig, axes = plt.subplots(ncols=2, figsize=(14, 4.5))
pc10 = explained.iloc[:, :10].stack().reset_index()
pc10.columns = ['Trial', 'Principal Component', 'Value']

pc10['Cumulative'] = pc10.groupby('Trial').Value.transform(np.cumsum)
print(pc10.info())
sns.barplot(x='Principal Component', y='Value', data=pc10, ax=axes[0])
sns.lineplot(x='Principal Component', y='Cumulative', data=pc10, ax=axes[1])
axes[1].set_xlim(1, 10)
axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
fig.suptitle('Explained Variance of Top 10 Principal Components | 100 Trials')
sns.despine()
fig.tight_layout()
fig.subplots_adjust(top=.90)
fig.savefig(plot_dataDir + 'Principle_components_100_trials.png')


# Searching for a weights to weigh the coins in a future portfolio.
# Visualizing the PCA factors in this way can also help decide which of the componenets to use
# As features in the dataset representing risk factors
# Read in MDF with initial coin pool
cpool_mdf = pd.read_csv(coin_dataDir + 'CoinPool.csv')
cpool_mdf.rename(columns={'Unnamed: 0': 'Dates'}, inplace=True)
cpool_mdf['Dates'] = pd.to_datetime(cpool_mdf['Dates'])
cpool_mdf.set_index(['Dates', 'Coin'], inplace=True)

top15 = cpool_mdf.loc['2022-01-07' , 'Market Cap'].nlargest(15)
top15 = top15.reset_index( level =1 )
top15.index = top15['Coin']
top15.drop('Coin',axis=1,inplace=True)
print(top15)

# Calculate the returns for  the top 15 coins based on Market Capitalization
returns_df = cpool_mdf.loc[idx[str(START):str(END), top15.index], 'Close'].unstack('Coin').pct_change()
print(returns_df.info())

# Winsorize at 2.5% and 97.5% quantiles
returns_df = returns_df.clip(lower=returns_df.quantile(q=.025),
                       upper=returns_df.quantile(q=.975),
                       axis=1)

# Base pool of top 15 by Mcap, for measures the Eigenportfolio performance at the end of the script
base_returns2 = returns_df

# Normalize/Scale
#new_df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)
normed_returns_df = pd.DataFrame(scale(returns_df
                       .clip(lower=returns_df.quantile(q=.025),
                             upper=returns_df.quantile(q=.975),
                             axis=1)
                      .apply(lambda x: x.sub(x.mean()).div(x.std()))), columns = returns_df.columns,index=returns_df.index)

#print(normed_returns_df.info())
#print(normed_returns_df.to_string())

# Drop coins and adates that do not have complete data for 95%  of the time period
normed_returns_df = normed_returns_df.dropna(thresh=int(normed_returns_df.shape[0] * .95), axis=1)
normed_returns_df = normed_returns_df.dropna(thresh=int(normed_returns_df.shape[1] * .95))

# To compare against wieghted Eigenportfolios at the end of the script
scaled_base_returns = normed_returns_df
print(normed_returns_df.info())
#print(normed_returns_df.to_string())

# Apply np.cov() to the normalized returns to see the strength of correlation among the coin returns
cov = normed_returns_df.cov()
covariance_map = sns.clustermap(cov)
covariance_map.savefig(plot_dataDir + 'Top_15_Covariance_Cluster_map.png')

# Feed the correlated returns to PCA and check which factors explain the most growth
pca = PCA()
pca = pca.fit(cov)
exp_var15 = pd.Series(pca.explained_variance_ratio_).to_frame('Explained Variance')
print(exp_var15.head().sum())
print(exp_var15.head())

# Normalize the four largest components PCA components so that they sum to 1
# Prepare to use them as weights for portfolios to compare to an equal-weighted portfolio formed from all coins
# In this case all the coins in the coin pool
top4 = pd.DataFrame(pca.components_[:4], columns=cov.columns)
eigen_portfolios = top4.div(top4.sum(1), axis=0)
eigen_portfolios.index = [f'Portfolio {i}' for i in range(1, 5)]


# Visualize the Eigenportfolio weights
axes = eigen_portfolios.T.plot.bar(subplots=True,
                                   layout=(2, 2),
                                   figsize=(14, 8),
                                   legend=False)
for ax in axes.flatten():
    ax.set_ylabel('Portfolio Weight')
    ax.set_xlabel('')
sns.despine()
plt.tight_layout()
plt.savefig(plot_dataDir + 'Eigenportfolio_weights.png')


# Visualize the performance of the Eigenportfolios
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 6), sharex=True)
axes = axes.flatten()
base_returns.mean(1).add(1).cumprod().sub(1).plot(title='The Market', ax=axes[0])
for i in range(3):
    rc = base_returns.mul(eigen_portfolios.iloc[i]).sum(1).add(1).cumprod().sub(1)
    rc.plot(title=f'Portfolio {i+1}', ax=axes[i+1], lw=1, rot=0)

for i in range(4):
    axes[i].set_xlabel('')
sns.despine()
fig.tight_layout()
fig.savefig(plot_dataDir + 'Eigenportfolio_performance.png')






