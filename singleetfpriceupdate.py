import yfinance as yf
import sqlite3
import pandas as pd
import datetime as dt
import argparse

ticker = '^892400-USD-STRD'

data = yf.download(ticker, interval='1d', start='2022-01-01', end='2025-09-15')
print("Downloaded data:")
print(data)
print("\nData columns:", data.columns)



database_path = 'foguth_etf_models.db'
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# --- FIX: Flatten the column index if it's a MultiIndex ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

data.reset_index(inplace=True)


# Remove rows with missing 'Close' data
data.dropna(inplace=True)

# create a dataframe out of the data
data['symbol'] = ticker
data = data[['Date', 'symbol', 'Close']]

# insert column etf_id and put the number 82 in every row of the dataframe
data['etf_id'] = 85
data = data[['Date', 'etf_id', 'symbol',  'Close']]

# insert the data into the etf_prices table
data.to_sql('etf_prices', conn, if_exists='append', index=False)
print(f"Inserted {len(data)} rows into the etf_prices table.")
conn.commit()
conn.close()
print("Data insertion complete.")