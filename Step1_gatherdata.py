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
# start = today - datetime.timedelta # would make it midnight yesterday morning
start = today
start = start.strftime('%s')  # make it epoch for the API

end = today - datetime.timedelta(90)  # get exactly 90 days worth of hourly data
end = end.strftime('%s')  # make it epoch for the API

# Fiat currency
fiat ='usd'


# Get the initial data. The first 100 coins listed on Coingecko.
# API call
response = requests.get(
    f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={fiat}&order=market_cap_desc&per_page=100&page=1&sparkline=false")
data = response.json()

print('Starting coin pool')
print(len(data))

# Drop coins with less than 2 Billion Market cap
for coin in data:
    if int(coin['market_cap']) < 2000000000:
        data.remove(coin)

# Drop less than 100 Million 24hr Volume (USD)
for coin in data:
    if int(coin['total_volume']) < 100000000:
        data.remove(coin)

# Drop stablecoins because its like having fiat in the data pool
stablecoins = ['usdt','busd',"ust",'mim','frax','tusd','usdc','dai']

for coin in data:
    for sc in stablecoins:
        if sc in coin['symbol']:
            data.remove(coin)

# Drop wrapped btc ERC20 Token, because its like having BTCx2 in the data pool
for coin in data:
    if 'wbtc' in coin['symbol']:
        data.remove(coin)

print('Coin pool after initial criteria')
print(len(data))
#pprint(data)

# Coins with a Mcap over USD 2 Billion and 24hr trade volume of over USD 100 Million
coinpool = [coin['id'] for coin in data]
tickers = [coin['symbol'].upper() for coin in data]
print(coinpool)
print(tickers)

# Custom function for getting Coingecko OHLCV data broken down by hour
def datagrabber(coin, fiat, start, end, ninetyDayPeriods):
    '''
    :param coin: name of cryptocoin according to coin 'id' on CoinGecko
    :param fiat:  name of a "regular" currency i.e. usd, eur, gbp and so on
    :param start: unix timestamp
    :param end: unix timestamp
    :param ninetyDayPeriods: number of 90 day periods. 90 is the maximum days that you can get hourly data for
    :return: A data frame of hourly OHLCV data for a cryptocoin(s)
    '''

    # API call for minutely data for the present day
    try:
        # API call, max 90 days for hourly prices. More than that its daily. API limitation
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range?vs_currency={fiat}&from={start}&to={now}")
        data = response.json()

    except:
        print("Couldn't connect to coingecko. Waiting 30 secs and trying again.....")
        time.sleep(30)
        # API call, max 90 days for hourly prices. More than that its daily. API limitation
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range?vs_currency={fiat}&from={start}&to={now}")
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

    # Getting more 90 day tranches of hourly OHLCV data. 4x90 = 360
    count = ninetyDayPeriods
    df_list = [todays_ohlc_df]


    while count > 0:
        print('Count is' + str(count) + f'for {coin}')

        start = df_list[0].index[0]
        end = start - datetime.timedelta(90)

        # Debug code
        #print('LOOK HERE FOR DATES') # Check this again once EDT returns
        #print(start)
        #print(end)

        start = start.strftime('%s')
        start = int(start) - 3600 # Temporary solution to a timezone issue likely realted to EDT/EST change
        start = str(start)

        end = end.strftime('%s')

        # Debug code
        #print('LOOK HERE FOR UNIX STAMPS')
        #print(start)
        #print(end)

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

        # Prepare the hourly price data
        new_df = pd.DataFrame(data['prices'], columns=['dates', 'prices'])
        new_df['dates'] = pd.to_datetime(new_df['dates'], unit='ms')
        new_df = new_df.set_index('dates')
        new_df.index = new_df.index.tz_localize('UTC').tz_convert(current_timezone)

        # Prepare hourly volume data
        volumes_df = pd.DataFrame(data['total_volumes'], columns=['dates', 'volumes'])
        volumes_df['dates'] = pd.to_datetime(volumes_df['dates'], unit='ms')
        volumes_df = volumes_df.set_index('dates')
        volumes_df.index = volumes_df.index.tz_localize('UTC').tz_convert(current_timezone)

        # Prepare hourly market cap data
        cap_df = pd.DataFrame(data['market_caps'], columns=['dates', 'mcaps'])
        cap_df['dates'] = pd.to_datetime(cap_df['dates'], unit='ms')
        cap_df = cap_df.set_index('dates')
        cap_df.index = cap_df.index.tz_localize('UTC').tz_convert(current_timezone)

        # Calculate the Open High Low Close Volume(OHLCV) values
        min_df = new_df.groupby([new_df.index.date]).min()  # Get the lows for each day
        max_df = new_df.groupby([new_df.index.date]).max()  # Get the highs for each day
        open_df = new_df.groupby([new_df.index.date]).first()  # Get the first value of each day for Open
        close_df = new_df.groupby([new_df.index.date]).last()  # Get the last value of each day for Close
        volume_df = volumes_df.groupby([volumes_df.index.date]).mean()  # Get the mean volume for each day
        mcaps_df = cap_df.groupby([cap_df.index.date]).mean()  # Get the mean mcaps for each day

        # Concatenate it all into a single dataframe for 90 days worth of OHLCV
        try:
            next_ohlc_df = pd.concat(
                [open_df['prices'], max_df['prices'], min_df['prices'], close_df['prices'], volume_df['volumes'],
                 mcaps_df['mcaps']], \
                axis=1, keys=['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'])

            df_list.insert(0, next_ohlc_df)
            count -= 1
            print('Pausing for 5 seconds, not to exceed API call limit')

            # Wait to not overload the API
            time.sleep(5)

            test_joiner = pd.concat(df_list)


        except KeyError:
            print(f'Something went wrong getting data for {coin}')
            print(f'Failure occured processing data for the dates from {end} to {start}')
            print(f'Skipping further loops for {coin}')


            test_joiner = pd.concat(df_list)


            break
    print(f"Finished for {coin}")
    return test_joiner

# Run the function. Get and save the data
for coin in coinpool:
    final_df = datagrabber(coin, fiat, start, end, 14)

    # Saves the plot graph to a file
    dataDir = 'DATA/COINHISTDATA/'
    isdir = os.path.isdir(dataDir)
    print(isdir)

    if isdir == False:
        os.makedirs(dataDir)
        print("Directory '% s' created" % dataDir)
        final_df.to_csv(f'{dataDir}{coin}.csv')
        print(f"Data for {coin} saved to {dataDir}")

    else:
        final_df.to_csv(f'{dataDir}{coin}.csv')
        print(f"Data for {coin} saved to {dataDir}")