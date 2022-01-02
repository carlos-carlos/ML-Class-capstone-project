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

# Global Settings

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

# Coin
coin = 'bitcoin'

# Data directories
coin_dataDir = 'DATA/COINHISTDATA/'

# Updater function
def update_data(coin,fiat,start,end):

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
    # print(new_df.to_string())

    # Prepare volume as we did before but this time its minutes of the present day, not the hours of the prior 90 days
    volumes_df = pd.DataFrame(data['total_volumes'], columns=['dates', 'volumes'])
    volumes_df['dates'] = pd.to_datetime(volumes_df['dates'], unit='ms')
    volumes_df = volumes_df.set_index('dates')
    volumes_df.index = volumes_df.index.tz_localize('UTC').tz_convert(current_timezone)
    # print(volumes_df.to_string())

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

    #ohlc_df = ohlc_df.append(todays_ohlc_df.iloc[-1], ignore_index=False)
    #print(todays_ohlc_df.to_string())
    #print(todays_ohlc_df.info)

    return todays_ohlc_df


# Get and prepare the existing data for the update
dir_list = os.listdir(coin_dataDir)
coins = [x.split('.')[0] for x in dir_list]

for coin in coins:
    # Read in current data
    current_df = pd.read_csv(coin_dataDir + f'{coin}.csv')
    current_df = current_df.loc[:, ~current_df.columns.str.contains('^Unnamed')]
    current_df = current_df.reindex(columns=["dates","open", "high", "low","close","volume"])
    current_df['dates'] = pd.to_datetime(current_df['dates'])
    current_df['dates'] = current_df['dates'].dt.normalize()

    # Get the the new days data
    update_df = update_data(coin,fiat,start,end)
    update_df = update_df.set_index('dates')

    print(update_df.to_string())
    print(update_df.info())

    # Sync up the columns and add the new day rows creating a new df in the process
    update_df.columns = current_df.columns.tolist()
    new_df = pd.concat([current_df, update_df]).drop_duplicates(subset=['dates'])

    print(new_df.to_string())
    print(new_df.info())