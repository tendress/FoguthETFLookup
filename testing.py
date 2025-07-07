import yfinance as yf
import sqlite3
import pandas as pd
import datetime as dt

# Sidebar for benchmark YTD performance
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
    #tickers = list(ticker_to_etf_id.keys())
    tickers = ['^GSPC', '^IXIC', '^DJI']
    try:
        all_data = yf.download(tickers, start='1994-01-01', end=enddate, interval="1d", group_by="ticker")
        if all_data.empty:
            print("No data returned for the tickers. Exiting...")
        else:
            #all_data.to_csv('all_data.csv')  # Save to CSV for debugging
            for ticker in tickers:
                if ticker in all_data.columns.levels[0]:
                    ticker_data = all_data[ticker].reset_index()
                    ticker_data['etf_id'] = ticker_to_etf_id[ticker]
                    ticker_data['symbol'] = ticker
                    ticker_data = ticker_data[['Date', 'etf_id', 'symbol', 'Close']]
                    ticker_data['Date'] = ticker_data['Date'].dt.strftime('%Y-%m-%d')
                    for _, row in ticker_data.iterrows():
                        cursor.execute('''
                            INSERT OR REPLACE INTO etf_prices (Date, etf_id, symbol, Close)
                            VALUES (?, ?, ?, ?)
                        ''', (row['Date'], row['etf_id'], row['symbol'], row['Close']))
                    print(f"Inserted new daily close price data for {ticker} into the etf_prices table.")
                else:
                    print(f"No data available for {ticker}. Skipping...")
    except Exception as e:
        print(f"Error fetching data: {e}")
    conn.commit()
    conn.close()




def main():
    database_path = 'foguth_etf_models.db'
    update_historical_prices(database_path)
    print("Historical prices updated successfully.")

if __name__ == "__main__":
    main()