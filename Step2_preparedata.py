import numpy as np
import pandas as pd
import sklearn as skl

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
print(coin_mdf.info())

# Sort index
coin_mdf.sort_index(inplace=True)
print(coin_mdf.to_string())

# Saves the Coin MDF to a file
dataDir = coinMDF_dataDir
isdir = os.path.isdir(dataDir)
print(isdir)

# Save the MDF in a seprate directory for persistence
if isdir == False:
    os.makedirs(dataDir)
    print("Directory '% s' created" % dataDir)
    coin_mdf.to_csv(f'{dataDir}CoinPool.csv')
    print(f"The initial pool of coins has been saved to {dataDir} as a MultiIndex dataframe")

else:
    coin_mdf.to_csv(f'{dataDir}CoinPool.csv')
    print(f"The initial pool of coins has been saved to {dataDir} as a MultiIndex dataframe")




