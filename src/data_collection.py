import yfinance as yf
import pandas as pd

def fetch_sp500_data(start_date='2010-01-01', end_date=None, save_path='sp500.csv'):
    ticker = '^GSPC'  # S&P 500 index symbol
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(save_path)
    print(f"S&P 500 data saved to {save_path}")

def fetch_forex_data(pair='EURUSD=X', start_date='2010-01-01', end_date=None, save_path='forex.csv'):
    data = yf.download(pair, start=start_date, end=end_date)
    data.to_csv(save_path)
    print(f"Forex data ({pair}) saved to {save_path}")

if __name__ == '__main__':
    fetch_sp500_data()
    fetch_forex_data() 