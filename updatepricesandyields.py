import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime

# make a variable for now using datetime module
now = datetime.now()

def fetch_etf_data(database_path):
    """
    Fetch a list of symbols from the etfs table and retrieve the current price and yield for each ETF.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the list of symbols from the etfs table
    cursor.execute('SELECT symbol FROM etfs')
    symbols = [row[0] for row in cursor.fetchall()]  # Extract symbols into a list
    print("Fetched symbols from the etfs table:", symbols)

    # Create a DataFrame to store the results
    etf_data = pd.DataFrame(columns=['Symbol', 'Current Price', 'Yield'])

    # Loop through each symbol and fetch data from yfinance
    for symbol in symbols:
        try:
            # Fetch data from yfinance
            etf = yf.Ticker(symbol)
            price = etf.history(period='1d')['Close'].iloc[-1] if not etf.history(period='1d').empty else None
            yield_info = etf.info.get('yield', None)
            etf_name = etf.info.get('longName', None)

            # Append the data to the DataFrame
            etf_data = pd.concat([etf_data, pd.DataFrame({
                'Symbol': [symbol],
                'Current Price': [price],
                'Yield': [yield_info],
                'Name': [etf_name]
            })], ignore_index=True)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            
            
    # update the etfs table with the fetched data and update the price_date
    for index, row in etf_data.iterrows():
        cursor.execute('''
            UPDATE etfs
            SET name = ?, price = ?, yield = ?, price_date = ?
            WHERE symbol = ?
        ''', (row['Name'], row['Current Price'], row['Yield'], now, row['Symbol']))     
    # Close the database connection
    conn.commit()
    conn.close()

    # Return the DataFrame
    return etf_data

# Path to the SQLite database
database_path = 'foguth_etf_models.db'

# Fetch ETF data and display it
etf_data = fetch_etf_data(database_path)
print("ETF Data:")
print(etf_data)