import yfinance as yf
import sqlite3
import pandas as pd
import datetime as dt

def getall_historical_prices(database_path, enddate=None):
    """
    Fetch historical daily price data for all tickers and store the 'Close' column in the etf_historical_prices table.
    """
    if enddate is None:
        enddate = dt.datetime.now().strftime('%Y-%m-%d')  # Default to today's date

    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create the etf_historical_prices table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS etf_historical_prices (
            Date TEXT NOT NULL,
            etf_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            Close REAL NOT NULL,
            PRIMARY KEY (Date, etf_id)
        )
    ''')
    conn.commit()

    # Fetch the list of tickers and their corresponding etf_id from the etfs table
    cursor.execute('SELECT id, symbol FROM etfs')
    etf_data = cursor.fetchall()  # Fetch all rows as a list of tuples
    print("List of tickers and their corresponding etf_id fetched from the etfs table:")
    print(etf_data)

    # Create a dictionary to map tickers to their etf_id
    ticker_to_etf_id = {ticker: etf_id for etf_id, ticker in etf_data}

    # Create a list of tickers
    tickers = list(ticker_to_etf_id.keys())

    # Fetch historical daily price data for all tickers in a single batch request
    try:
        print("Fetching historical data for all tickers...")
        all_data = yf.download(tickers, end=enddate, interval="1d", group_by="ticker")
        print(all_data)

        # Process the data to extract the 'Close' column and filter out NaN values
        for ticker in tickers:
            if ticker in all_data.columns.levels[0]:  # Ensure data exists for the ticker
                ticker_data = all_data[ticker]['Close'].reset_index()  # Get the 'Close' column and reset the index
                ticker_data = ticker_data.dropna(subset=['Close'])  # Drop rows where 'Close' is NaN
                ticker_data['etf_id'] = ticker_to_etf_id[ticker]  # Add the etf_id to the DataFrame
                ticker_data['symbol'] = ticker  # Add the ticker symbol to the DataFrame
                ticker_data['Date'] = ticker_data['Date'].dt.strftime('%Y-%m-%d')  # Format the date as a string

                # Insert data into the etf_historical_prices table
                for _, row in ticker_data.iterrows():
                    cursor.execute('''
                        INSERT OR IGNORE INTO etf_historical_prices (Date, etf_id, symbol, Close)
                        VALUES (?, ?, ?, ?)
                    ''', (row['Date'], row['etf_id'], row['symbol'], row['Close']))

                print(f"Inserted historical price data for {ticker} into the etf_historical_prices table.")
            else:
                print(f"No data available for {ticker}. Skipping...")

        conn.commit()

    except Exception as e:
        print(f"Error fetching data: {e}")

    finally:
        conn.close()

# Example usage
database_path = 'foguth_etf_models.db'
getall_historical_prices(database_path)