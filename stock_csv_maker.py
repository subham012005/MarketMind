from curl_cffi import requests
import yfinance as yf
from stock_csv_better_maker import convert_stock_data
import time

def download_stock_data(ticker, start_date, end_date):
    session = requests.Session(impersonate="chrome")
    stock = yf.Ticker(ticker, session=session)
    data = stock.history(start=start_date, end=end_date)
    raw_file = f"{ticker}_historical_data.csv"
    cleaned_file = f"cleaned_{ticker}_historical_data.csv"

    data.to_csv(raw_file)
    print(f"Data saved as {raw_file}")

    time.sleep(5)  # Optional wait to simulate delay

    convert_stock_data(raw_file, cleaned_file)

    return cleaned_file
