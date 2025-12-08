import sqlite3
import pandas as pd
import yfinance as yf
import datetime as dt

DB_PATH = 'foguth_etf_models.db'
TICKERS = ['XLE', 'XLU', 'XLK']
START_DATE = '1994-01-01'
END_DATE = dt.datetime.now().strftime('%Y-%m-%d')

def delete_target_rows(conn, tickers):
    cur = conn.cursor()
    cur.execute(f"DELETE FROM etf_prices WHERE symbol IN ({','.join(['?']*len(tickers))})", tickers)
    conn.commit()

def map_tickers_to_ids(conn, tickers):
    cur = conn.cursor()
    cur.execute(f"SELECT id, symbol FROM etfs WHERE symbol IN ({','.join(['?']*len(tickers))})", tickers)
    rows = cur.fetchall()
    mapping = {sym: etf_id for etf_id, sym in rows}
    missing = [t for t in tickers if t not in mapping]
    if missing:
        raise RuntimeError(f"Missing symbols in etfs table: {missing}")
    return mapping

def download_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", group_by="ticker")
    if data.empty:
        raise RuntimeError("yfinance returned no data.")
    return data

def rebuild(conn, all_data, mapping):
    cur = conn.cursor()
    rows = []
    for ticker in mapping.keys():
        if isinstance(all_data.columns, pd.MultiIndex):
            if ticker not in all_data.columns.levels[0]:
                print(f"No data returned for {ticker}, skipping.")
                continue
            df = all_data[ticker].reset_index()
        else:
            df = all_data.reset_index()
        if 'Close' not in df.columns:
            print(f"'Close' column missing for {ticker}, skipping.")
            continue
        df = df.dropna(subset=['Close']).copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['symbol'] = ticker
        df['etf_id'] = mapping[ticker]
        rows.extend(df[['Date','etf_id','symbol','Close']].itertuples(index=False, name=None))

    if not rows:
        print("No rows prepared for insert.")
        return

    cur.execute("BEGIN IMMEDIATE")
    cur.executemany("""
        INSERT INTO etf_prices (Date, etf_id, symbol, Close)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(Date, etf_id) DO UPDATE SET
            symbol = excluded.symbol,
            Close  = excluded.Close
    """, rows)
    conn.commit()
    print(f"Rebuilt etf_prices for {len(mapping)} tickers. Inserted/updated {len(rows)} rows.")

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        delete_target_rows(conn, TICKERS)
        mapping = map_tickers_to_ids(conn, TICKERS)
        all_data = download_prices(TICKERS, START_DATE, END_DATE)
        rebuild(conn, all_data, mapping)
    finally:
        conn.close()

if __name__ == "__main__":
    main()