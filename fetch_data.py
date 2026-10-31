#This is the file for data extraction and preprocessing.
#First, the data will be extracted from the yfinance library.
import yfinance as yf
import pandas as pd

stocks = ["RELIANCE.NS", "TCS.NS", "ITC.NS", "MARUTI.NS", "HDFCBANK.NS"]

data = yf.download(stocks, start = "2008-01-07", end = "2026-10-31", threads = False)

data.to_csv("raw_data.csv")
print("Download complete, saved into raw_data.csv")