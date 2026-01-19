import yfinance as yf
import sqlite3
import pandas as pd

# --- Identifiers you can edit ---
ETF_ID = 89            # the etf_id stored in your DB
SYMBOL = 'PJFV'        # the DB symbol to store
TICKER = 'PJFV'        # the Yahoo Finance ticker to download (can match SYMBOL)
START_DATE = '2024-11-26'  # adjust as needed
# -------------------------------

data = yf.download(TICKER, interval='1d', start=START_DATE)
print("Downloaded data:")
print(data)
print("\nData columns:", data.columns)

database_path = 'foguth_etf_models.db'
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Flatten MultiIndex if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Ensure a clean Date + Close shape
data = data.reset_index()
if 'Date' not in data.columns:  # safety if index wasn't named
    data.rename(columns={'index': 'Date'}, inplace=True)

# Keep only rows with a Close value
data = data.dropna(subset=['Close']).copy()

# Add identifiers and normalize types
data['symbol'] = SYMBOL
data['etf_id'] = ETF_ID
data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Remove any duplicates within this batch
data = data.drop_duplicates(subset=['Date', 'etf_id'])

# Filter out rows already in the table for this etf_id
existing = pd.read_sql_query(
    "SELECT Date FROM etf_prices WHERE etf_id = ?",
    conn, params=(ETF_ID,)
)
existing_dates = set(existing['Date'].astype(str))
data = data[~data['Date'].isin(existing_dates)]

# Nothing new? Exit gracefully
if data.empty:
    print("No new rows to insert for this ticker/etf_id.")
    conn.close()
else:
    rows = list(data[['Date', 'etf_id', 'symbol', 'Close']].itertuples(index=False, name=None))

    # UPSERT: update Close/symbol if the (Date, etf_id) row exists
    cursor.executemany("""
        INSERT INTO etf_prices (Date, etf_id, symbol, Close)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(Date, etf_id) DO UPDATE SET
            symbol = excluded.symbol,
            Close  = excluded.Close
    """, rows)

    # If your SQLite is too old for ON CONFLICT DO UPDATE, use IGNORE instead:
    # cursor.executemany("""
    #     INSERT OR IGNORE INTO etf_prices (Date, etf_id, symbol, Close)
    #     VALUES (?, ?, ?, ?)
    # """, rows)

    conn.commit()
    print(f"Upserted {len(rows)} rows into etf_prices.")
    conn.close()