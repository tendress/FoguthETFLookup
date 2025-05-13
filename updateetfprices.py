import yfinance as yf
import sqlite3
import pandas as pd
import datetime as dt

def update_historical_prices(database_path, enddate=None):
    """
    Fetch historical daily price data for all tickers and store it in the etf_prices table.
    """
    if enddate is None:
        enddate = dt.datetime.now().strftime('%Y-%m-%d')  # Default to today's date

    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

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
        all_data = yf.download(tickers,  end=enddate, interval="1d", group_by="ticker")

        # Check if data is returned
        if all_data.empty:
            print("No data returned for the tickers. Exiting...")
        else:
            # Process data for each ticker
            # Process data for each ticker
            for ticker in tickers:
                if ticker in all_data.columns.levels[0]:  # Ensure data exists for the ticker
                    ticker_data = all_data[ticker].reset_index()  # Reset index to access the Date column
                    ticker_data['etf_id'] = ticker_to_etf_id[ticker]  # Add the etf_id to the DataFrame
                    ticker_data['symbol'] = ticker  # Add the ticker symbol to the DataFrame
                    ticker_data = ticker_data[['Date', 'etf_id', 'symbol', 'Close']]  # Select only Date, etf_id, symbol, and Close columns
                    ticker_data['Date'] = ticker_data['Date'].dt.strftime('%Y-%m-%d')  # Format the date as a string

                    # Insert data into the etf_prices table
                    for _, row in ticker_data.iterrows():
                        cursor.execute('''
                            INSERT OR IGNORE INTO etf_prices (Date, etf_id, symbol, Close)
                            VALUES (?, ?, ?, ?)
                        ''', (row['Date'], row['etf_id'], row['symbol'], row['Close']))

                    print(f"Inserted new daily close price data for {ticker} into the etf_prices table.")
                else:
                    print(f"No data available for {ticker}. Skipping...")
    except Exception as e:
        print(f"Error fetching data: {e}")

    # Close the database connection
    conn.close()
    print("Daily close price data has been stored in the etf_prices table.")

def update_latest_prices(database_path):
    """
    Fetch the most recent price for all tickers and update or overwrite the current value for this date in the etf_prices table.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the list of tickers and their corresponding etf_id from the etfs table
    cursor.execute('SELECT id, symbol FROM etfs')
    etf_data = cursor.fetchall()  # Fetch all rows as a list of tuples
    print("List of tickers and their corresponding etf_id fetched from the etfs table:")
    print(etf_data)

    # Create a dictionary to map tickers to their etf_id
    ticker_to_etf_id = {ticker: etf_id for etf_id, ticker in etf_data}

    # Create a list of tickers
    tickers = list(ticker_to_etf_id.keys())

    # Fetch the most recent price for all tickers
    try:
        print("Fetching the most recent price for all tickers...")
        for ticker in tickers:
            # Fetch intraday data for the current day
            ticker_data = yf.Ticker(ticker).history(period="1d", interval="1m")

            # Check if data is returned
            if ticker_data.empty:
                print(f"No data returned for {ticker}. Skipping...")
                continue

            # Get the most recent price
            latest_row = ticker_data.iloc[-1]
            latest_price = latest_row['Close']
            latest_date = dt.datetime.now().strftime('%Y-%m-%d')  # Use today's date

            # Update or insert the most recent price into the etf_prices table
            cursor.execute('''
                INSERT INTO etf_prices (Date, etf_id, symbol, Close)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(Date, etf_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    Close = excluded.Close
            ''', (latest_date, ticker_to_etf_id[ticker], ticker, latest_price))
            print(f"Updated the most recent price for {ticker} in the etf_prices table.")
    except Exception as e:
        print(f"Error fetching data: {e}")

    # Commit changes and close the database connection
    conn.commit()
    conn.close()
    print("Most recent price data has been updated in the etf_prices table.")

if __name__ == "__main__":
    # Example usage
    database_path = 'foguth_etf_models.db'  # Path to your SQLite database
    update_historical_prices(database_path)
    update_latest_prices(database_path)