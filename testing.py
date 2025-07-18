import yfinance as yf
import sqlite3
import pandas as pd
import datetime as dt

def update_historical_prices(database_path, enddate=None):
    """
    Fetch historical daily price data for all tickers and store it in the etf_prices table.
    """
    if enddate is None:
        enddate = dt.datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, symbol FROM etfs')
    etf_data = cursor.fetchall()
    ticker_to_etf_id = {ticker: etf_id for etf_id, ticker in etf_data}
    tickers = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^IRX', '^FVX', '^TNX', '^TYX']
    
    try:
        all_data = yf.download(tickers, start='1994-01-01', end=enddate, interval="1d", group_by="ticker")
        if all_data.empty:
            print("No data returned for the tickers. Exiting...")
        else:
            for ticker in tickers:
                if ticker in all_data.columns.levels[0]:
                    ticker_data = all_data[ticker].reset_index()
                    
                    # Filter out rows where Close is NaN or None
                    ticker_data = ticker_data.dropna(subset=['Close'])
                    
                    if ticker_data.empty:
                        print(f"No valid Close data for {ticker}. Skipping...")
                        continue
                    
                    ticker_data['etf_id'] = ticker_to_etf_id[ticker]
                    ticker_data['symbol'] = ticker
                    ticker_data = ticker_data[['Date', 'etf_id', 'symbol', 'Close']]
                    ticker_data['Date'] = ticker_data['Date'].dt.strftime('%Y-%m-%d')
                    
                    rows_inserted = 0
                    for _, row in ticker_data.iterrows():
                        # Double-check that Close is not NaN before inserting
                        if pd.notna(row['Close']) and row['Close'] is not None:
                            cursor.execute('''
                                INSERT OR REPLACE INTO etf_prices (Date, etf_id, symbol, Close)
                                VALUES (?, ?, ?, ?)
                            ''', (row['Date'], row['etf_id'], row['symbol'], row['Close']))
                            rows_inserted += 1
                    
                    print(f"Inserted {rows_inserted} valid rows for {ticker}")
                else:
                    print(f"No data available for {ticker}. Skipping...")
    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()
    
    conn.commit()
    conn.close()

def main():
    database_path = 'foguth_etf_models.db'
    update_historical_prices(database_path)
    print("Historical prices updated successfully.")

if __name__ == "__main__":
    main()