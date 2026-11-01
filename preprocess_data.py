import pandas as pd
import numpy as np
#Using the dates column as the index. Also, header = [0,1] is used here as the data of the top 2 rows was very clunky to process, so the price(open, high, low, close) along with the tickers were made the column headings.
df = pd.read_csv("raw_data.csv", header=[0,1], index_col = 0, parse_dates=True)
#This is a 25 columns dataframe, with 5 prices for 5 tickers. So, using stack, this will be compressed into 5 columns. level = 1 wil be used to transfer the ticker as a column. Also, this will result in dates being repeated(each date 5 times), so reset index.
df = df.stack(level=1, future_stack=True).reset_index()
#Now, there were some dates where there was an exchange holiday. yfinance padded those rows as open=high=low=close, with volume = 0. Those rows will be filtered out. 
df = df[df['Volume']>0]
df = df.sort_values(by = ['Ticker', 'Date']).reset_index(drop=True) #index rebuilt to account for dropped rows.
#Daily log return = ln(Ct/Ct-1)
df['DailyLogReturn'] = np.log(df['Close'] / df.groupby('Ticker')['Close'].shift(1))
df['IntradayRange'] = (df['High']-df['Low'])/df['Close']
#Standard deviation over the past few days(volatility), multiplied by square root of 252(number of trading days in a year), to change the unit to yearly.
df['Vol_5d'] = df.groupby('Ticker')['DailyLogReturn'].transform(
    lambda x: x.rolling(5).std(ddof=1) * np.sqrt(252)
)
df['Vol_21d'] = df.groupby('Ticker')['DailyLogReturn'].transform(
    lambda x: x.rolling(21).std(ddof=1) * np.sqrt(252)
)
#Ratio of mean of volume over the past 3 days to that over the past 21 days.
df['VolumeRatio'] = df.groupby('Ticker')['Volume'].transform(
    lambda x: x.rolling(3).mean()/x.rolling(21).mean()
)
#Cumulative return over the past 5 days, so as to get an idea of whether trend of daily returns is positive or negative.
df['5dayCumReturn'] = df.groupby('Ticker')['DailyLogReturn'].transform(
    lambda x: x.rolling(5).sum()
)
#This is the target columns, the volatility over the next 5 days.
df['Target_Vol_5d'] = df.groupby('Ticker')['DailyLogReturn'].transform(
    lambda x: (x.rolling(5).std(ddof=1) * np.sqrt(252)).shift(-5)
)
#The above calculations have NaN'ed many edge cases. They must be dealt with.
df = df.dropna().reset_index(drop=True)
#Only the columns concerning our features and target will be kept with.
cols_to_keep = [
    'Date', 'Ticker', 'DailyLogReturn', '5dayCumReturn',
    'Vol_5d', 'Vol_21d', 'IntradayRange', 'VolumeRatio',
    'Target_Vol_5d'
]
df = df[cols_to_keep]
df.to_csv("processed_data.csv", index = False)