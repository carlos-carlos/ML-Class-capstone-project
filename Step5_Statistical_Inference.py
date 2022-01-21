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

print('Initial Base Data:')
print(model_mdf.info())

# Drop NaNs and the OHLCV columns
data = (model_mdf
            .dropna()
            .drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1))

print(data.info(show_counts=True))

# Ready the data for the model
y = data.filter(like='target')
X = data.drop(y.columns, axis=1)

print(X.info(show_counts=True))
print(y.info(show_counts=True))

#check_nan = X.isnull().values.any()
#print(check_nan)

# Clustermap Divergence Pallete
j = sns.clustermap(y.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0, annot=True, fmt='.2%')
j.savefig(plot_dataDir + 'Returns_Divergence_ClusterMap.png')


j = sns.clustermap(X.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0)
j.gcf().set_size_inches((14, 14))
j.savefig(plot_dataDir + 'Features_Divergence_ClusterMap.png')
