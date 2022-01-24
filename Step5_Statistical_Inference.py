import pandas as pd

from statsmodels.api import OLS, add_constant, graphics
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt

# GLOBAL SETTINGS

# Data directories

#coin_dataDir = 'DATA/TESTDIR/' # Debug dir for testing I/O logic and/or issues. It should be a clone of the above dir.
model_dataDir = 'DATA/MODELDATA/'
plot_dataDir = 'DATA/INITIAL_INSIGHTS/MOMENTUM_FACTORS/STATINFER/'


# Helpers
idx = pd.IndexSlice
sns.set_style('whitegrid')

# END GLOBAL SETTINGS

# Read in MDF with initial coin pool
model_mdf = pd.read_csv(model_dataDir + 'ModelData.csv')
model_mdf.rename(columns={'Unnamed: 0': 'Dates'}, inplace=True)
model_mdf['Dates'] = pd.to_datetime(model_mdf['Dates'])
model_mdf.set_index(['Dates', 'Coin'], inplace=True)

#print('Initial Base Data:')
#print(model_mdf.info())

# Drop NaNs and the OHLCV columns
data = (model_mdf
            .dropna()
            .drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1))

#print(data.info(show_counts=True))

# Ready the data for the model
y = data.filter(like='target')
X = data.drop(y.columns, axis=1)

X.loc[:, 'year_2021':'month_12'] = X.loc[:, 'year_2021':'month_12'].astype('uint8')
#print(X.info(show_counts=True))

#print(y.info(show_counts=True))

'''
check_nan = X.isnull().values.any()
#print(check_nan)
#print(X.eq('NaN').values.any())
#print(X.empty)
#print(X.notnull().values.any())
#print(X.info())
#print(X.describe())
#print(X.shape)



for c in X.columns:
    print(c)
    X.loc[X[c] == 1, c] = 2
    X.loc[X[c] == 0, c] = 1

print(X.to_string())

X.loc[:, 'year_2021':'month_12'] = X.loc[:, 'year_2021':'month_12'].astype('uint8')
print(X.info(show_counts=True))
#print(X.to_string())
'''

# Month and year features causing infinite value error with the cluster map.
# While debugging this issue, X2 is a copy of X with the time factors removed, so the clustermap
# Does not crash the program.

X2 = X

mon = X2.filter(like='month')
X2 = X2.drop(mon.columns,axis=1)
#print(X2.info(show_counts=True))
#print(mon.to_string())


mon = X2.filter(like='year')
X2 = X2.drop(mon.columns,axis=1)
#print(X2.info(show_counts=True))
#print(mon.to_string())


# Clustermap Divergence Pallete for y
#j = sns.clustermap(y.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0, annot=True, fmt='.2%')
#j.savefig(plot_dataDir + 'Returns_Divergence_ClusterMap.png')

# Clustermap Divergence Pallete for X (X2)
#j = sns.clustermap(X2.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0)
#plt.gcf().set_size_inches((14, 14))
#j.savefig(plot_dataDir + 'Features_Divergence_ClusterMap.png')

# Correlation matrix
print("FEATURE CORRELATION:")
corr_mat = X.corr().stack().reset_index()
corr_mat.columns=['var1', 'var2', 'corr']
corr_mat = corr_mat[corr_mat.var1!=corr_mat.var2].sort_values(by='corr', ascending=False)
print(corr_mat.head().append(corr_mat.tail()))

# Target value Boxplot
target_box = y.boxplot()
#fig = target_box.get_figure()
#fig.savefig(plot_dataDir + 'Target_Y_Boxplot.png')

# Statistical Inference with OLS from statsmodels

# Standardize the data by coin
X = (X.groupby(level='Coin')
     .transform(lambda x: (x - x.mean()) / x.std())
    .fillna(0))

# OLS against Daily targets
target = 'target_1d'
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()

print(f"OLS Regression results with {target} as the Y Target values".upper())
print(trained_model.summary())

# Obtaining residuals
preds = trained_model.predict(add_constant(X))
residuals = y[target] - preds

fig, axes = plt.subplots(ncols=2, figsize=(14,4))
res_plot = sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title(f'Residual Distribution {target}')
axes[0].legend()
plot_acf(residuals, lags=40, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout()
fig.savefig(plot_dataDir + f'{target}_Residuals.png')


# OLS against Weekly targets
target = 'target_7d'
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()

print(f"OLS Regression results with {target} as the Y Target values".upper())
print(trained_model.summary())

# Obtaining residuals
preds = trained_model.predict(add_constant(X))
residuals = y[target] - preds

fig, axes = plt.subplots(ncols=2, figsize=(14,4))
res_plot = sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title(f'Residual Distribution {target}')
axes[0].legend()
plot_acf(residuals, lags=40, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout()
fig.savefig(plot_dataDir + f'{target}_Residuals.png')


# OLS against bi-weekly targets
target = 'target_14d'
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()

print(f"OLS Regression results with {target} as the Y Target values".upper())
print(trained_model.summary())

# Obtaining residuals
preds = trained_model.predict(add_constant(X))
residuals = y[target] - preds

fig, axes = plt.subplots(ncols=2, figsize=(14,4))
res_plot = sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title(f'Residual Distribution {target}')
axes[0].legend()
plot_acf(residuals, lags=50, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout()
fig.savefig(plot_dataDir + f'{target}_Residuals.png')


# OLS against monthly targets
target = 'target_30d'
model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()

print(f"OLS Regression results with {target} as the Y Target values".upper())
print(trained_model.summary())

# Obtaining residuals
preds = trained_model.predict(add_constant(X))
residuals = y[target] - preds

fig, axes = plt.subplots(ncols=2, figsize=(14,4))
res_plot = sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title(f'Residual Distribution {target}')
axes[0].legend()
plot_acf(residuals, lags=50, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout()
fig.savefig(plot_dataDir + f'{target}_Residuals.png')
