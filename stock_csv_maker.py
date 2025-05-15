import yfinance as yf
from stock_csv_better_maker import clean_multirow_csv
import time

def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    raw_file = f"{ticker}_historical_data.csv"
    cleaned_file = f"cleaned_{ticker}_historical_data.csv"

    data.to_csv(raw_file)
    print(f"Data saved as {raw_file}")

    time.sleep(5)  # Optional wait to simulate delay

    clean_multirow_csv(raw_file, cleaned_file)

    return cleaned_file
download_stock_data("AAPL", "2020-01-01", "2023-10-01")