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
# Time controls
timezone = 'US/Eastern'
current_timezone = pytz.timezone(timezone)

now = datetime.datetime.now()  # current date and time
today = datetime.datetime.combine(now, datetime.datetime.min.time())  # current data, midnight this morning UTC
now = now.strftime('%s')  # make this epoch time but after you used it for the above purpose

# Start and end times for API call in epoch
#start = today - datetime.timedelta # would make it midnight yesterday morning
start = today
start = start.strftime('%s')  # make it epoch for the API
start = int(start) - 3600  # Temporary solution to a timezone issue likely realted to EDT/EST change

end = today - datetime.timedelta(90)  # get exactly 90 days worth of hourly data
end = end.strftime('%s')  # make it epoch for the API

# Fiat currency
fiat ='usd'

# Data directories
coin_dataDir = 'DATA/COINHISTDATA/'
#coin_dataDir = 'DATA/TESTDIR/HIST/' # Debug dir for testing I/O logic and/or issues. It should be a clone of the above dir.
coinMDF_dataDir = 'DATA/COMBINEDDATA/'
#coinMDF_dataDir = 'DATA/TESTDIR/POOL/' # Debug dir for testing. It should be a clone of the above dir.

# Helpers
idx = pd.IndexSlice

# END GLOBAL SETTINGS

# FUNCTIONS
# Updater function
def update_data(coin,fiat,start,end):
    '''
    :param coin: Str: Cryptocoin name
    :param fiat: Str: USD,GBP,EUR,JPY etc..
    :param start: UNIX timestamp in str form
    :param end: UNIX timestamp in str form
    :return: The original data with new days added from the most recent date to yesterday
    '''

    # API call for minutely data for the present day
    try:
        # API call, max 90 days for hourly prices. More than that its daily. API limitation
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range?vs_currency={fiat}&from={end}&to={start}")
        data = response.json()

    except:
        print("Couldn't connect to coingecko. Waiting 30 secs and trying again.....")
        time.sleep(30)
        # API call, max 90 days for hourly prices. More than that its daily. API limitation
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range?vs_currency={fiat}&from={end}&to={start}")
        data = response.json()


    # Prepare prices as we did before but this time its minutes of the present day, not the hours of the prior 90 days
    new_df = pd.DataFrame(data['prices'], columns=['dates', 'prices'])
    new_df['dates'] = pd.to_datetime(new_df['dates'], unit='ms')
    new_df = new_df.set_index('dates')
    new_df.index = new_df.index.tz_localize('UTC').tz_convert(current_timezone)

    # Prepare volume as we did before but this time its minutes of the present day, not the hours of the prior 90 days
    volumes_df = pd.DataFrame(data['total_volumes'], columns=['dates', 'volumes'])
    volumes_df['dates'] = pd.to_datetime(volumes_df['dates'], unit='ms')
    volumes_df = volumes_df.set_index('dates')
    volumes_df.index = volumes_df.index.tz_localize('UTC').tz_convert(current_timezone)

    # Prepare hourly market cap data
    cap_df = pd.DataFrame(data['market_caps'], columns=['dates', 'mcaps'])
    cap_df['dates'] = pd.to_datetime(cap_df['dates'], unit='ms')
    cap_df = cap_df.set_index('dates')
    cap_df.index = cap_df.index.tz_localize('UTC').tz_convert(current_timezone)

    # Calculate todays Open High Low Close Volume(OHLCV) values
    min_df = new_df.groupby([new_df.index.date]).min()  # Get the lows for each day
    max_df = new_df.groupby([new_df.index.date]).max()  # Get the highs for each day
    open_df = new_df.groupby([new_df.index.date]).first()  # Get the first value of each day for Open
    close_df = new_df.groupby([new_df.index.date]).last()  # Get the last value of each day for Close
    volume_df = volumes_df.groupby([volumes_df.index.date]).last()  # Get the latest volume for the day
    mcaps_df = cap_df.groupby([cap_df.index.date]).mean()  # Get the mean mcaps for each day

    todays_ohlc_df = pd.concat(
        [open_df['prices'], max_df['prices'], min_df['prices'], close_df['prices'], volume_df['volumes'],
         mcaps_df['mcaps']], \
        axis=1, keys=['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'])

    # Debug code
    #print(todays_ohlc_df.to_string())
    #print(todays_ohlc_df.info)

    return todays_ohlc_df

# Filter function
def filterdata(mdf, rule, window, threshold):
    '''
    :param mdf: A MultiIndex DataFrame
    :param rule: Str 'Market Cap' or 'Volume'
    :param window: Int: the window for the rolling mean to compare against threshold
    :param threshold: Int: The cutoff to decide what gets filtered
    :return: MultiIndex Dataframe filtered by the abpve
    '''
    filtered = []
    for coin, new_df in mdf.groupby(level=1):
        if new_df.set_index(new_df.index.get_level_values('Dates'))[f'{rule}'].rolling(window).mean().mean() > threshold:
            filtered.append(new_df)

    mdf = pd.concat(filtered)
    mdf.sort_index(inplace=True)

    return mdf

# END FUNCTIONS

# START UPDATE LOGIC

# Decide each time whether to update the data or use the existing local data
# Might want to do this if offline or if you are debugging and don't want to slam the API each time you run the script.
print('Would you like to update the local data?')
should_update = input('Enter "y" if YES. Enter anything else or nothing if NO').lower()
should_filter = input('Enter "y" if YES. Enter anything else or nothing if NO').lower()

# If you update the data
if should_update == 'y':
    # Get and prepare the existing data for the update
    dir_list = os.listdir(coin_dataDir)
    coins = [x.split('.')[0] for x in dir_list]

    # Update the data
    for coin in coins:
        # Read in current data
        current_df = pd.read_csv(coin_dataDir + f'{coin}.csv')
        current_df.rename(columns={'Unnamed: 0': 'dates'}, inplace=True)
        current_df = current_df.set_index('dates')
        current_df.drop(current_df.index[-1], inplace=True)
        current_df['dates'] = current_df.index

        # Get the the new days data
        update_df = update_data(coin, fiat, start, end)
        update_df["dates"] = update_df.index

        # Combine the 2 dataframes
        new_df = pd.concat([current_df, update_df])
        new_df['dates'] = pd.to_datetime(new_df['dates'])
        new_df = new_df.drop_duplicates(subset=["dates"])
        new_df = new_df.drop(['dates'], axis=1)

        # Debug code
        # print('THE DATA AFTER THE UPDATE')
        # print(new_df.to_string())
        # print(new_df.info())

        # Save the updated
        new_df.to_csv(f'{coin_dataDir}{coin}.csv')
        print(f"Data for {coin} saved to {coin_dataDir}")

        # END UPDATE LOGIC

    # Create a single pool of coins

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
    # print(coin_mdf.info())

    # Sort index
    coin_mdf.sort_index(inplace=True)
    #print(coin_mdf.to_string())
    #print(coin_mdf.loc[('2022-01-02', 'bitcoin')]['Close'])
    #print(coin_mdf.index)
    if should_filter == 'y':
        coin_mdf = filterdata(coin_mdf, 'Market Cap', 180, 10000000000)

    #print(coin_mdf.to_string())


    # Saves the Coin MDF to a file
    dataDir = coinMDF_dataDir
    isdir = os.path.isdir(dataDir)

    # Save the initial pool MDF in a seprate directory for persistence and "just in case" purposes
    if isdir == False:
        os.makedirs(coinMDF_dataDir)
        print("Directory '% s' created" % coinMDF_dataDir)
        coin_mdf.to_csv(f'{coinMDF_dataDir}CoinPool.csv')
        print(f"The initial pool of coins has been saved to {coinMDF_dataDir} as a MultiIndex dataframe")

    else:
        coin_mdf.to_csv(f'{coinMDF_dataDir}CoinPool.csv')
        print(f"The initial pool of coins has been saved to {coinMDF_dataDir} as a MultiIndex dataframe")

# If you do not update the data
else:
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

    if should_filter == 'y':
        coin_mdf = filterdata(coin_mdf, 'Market Cap', 180, 10000000000)

    #print(coin_mdf.to_string())

    # Saves the Coin MDF to a file
    dataDir = coinMDF_dataDir
    isdir = os.path.isdir(dataDir)

    # Save the initial pool MDF in a seprate directory for persistence and "just in case" purposes
    if isdir == False:
        os.makedirs(coinMDF_dataDir)
        print("Directory '% s' created" % coinMDF_dataDir)
        coin_mdf.to_csv(f'{coinMDF_dataDir}CoinPool.csv')
        print(f"The initial pool of coins has been saved to {coinMDF_dataDir} as a MultiIndex dataframe")

    else:
        coin_mdf.to_csv(f'{coinMDF_dataDir}CoinPool.csv')
        print(f"The initial pool of coins has been saved to {coinMDF_dataDir} as a MultiIndex dataframe")


