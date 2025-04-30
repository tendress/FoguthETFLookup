import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime

def update_etf_ytd_returns(database_path):
    """
    Update the YTD returns for ETFs in the database.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the list of tickers from the etfs table
    cursor.execute('SELECT symbol FROM etfs')
    myTickers = [row[0] for row in cursor.fetchall()]  # Create a list of tickers
    print("List of tickers fetched from the etfs table:")
    print(myTickers)

    # Create a DataFrame to store the data
    df = pd.DataFrame()

    # Loop through each ticker and get the yield, price, price_date, and YTD price return
    for ticker in myTickers:
        new_ticker = yf.Ticker(ticker)
        try:
            # Fetch yield information
            yield_info = new_ticker.info.get('yield', None)

            # Fetch the latest close price
            price_info = new_ticker.history(period='1d')['Close'].iloc[-1] if not new_ticker.history(period='1d').empty else None

            # Get the current date for the price_date column
            price_date = datetime.now().strftime('%Y-%m-%d')

            # Fetch year-to-date price history
            ytd_data = new_ticker.history(period='ytd')
            if not ytd_data.empty:
                start_price = ytd_data['Close'].iloc[0]  # Price on the first trading day of the year
                ytd_price_return = ((price_info - start_price) / start_price) * 100 if start_price else None
            else:
                ytd_price_return = None

            # Append the data to the DataFrame
            df = pd.concat([df, pd.DataFrame({
                'Ticker': [ticker],
                'Yield': [yield_info],
                'Price': [price_info],
                'PriceDate': [price_date],
                'YTDPriceReturn': [ytd_price_return]
            })], ignore_index=True)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # Sort the DataFrame by YTD price return in descending order
    df = df.sort_values(by='YTDPriceReturn', ascending=False).reset_index(drop=True)
    print("DataFrame with yield, price, price_date, and YTD price return information:")
    print(df)

    # Update the 'yield', 'price', 'price_date', 'YTDPriceReturn', and 'PriceReturnDate' columns in the etfs table
    for index, row in df.iterrows():
        cursor.execute('''
            UPDATE etfs
            SET yield = ?, price = ?, price_date = ?, YTDPriceReturn = ?, PriceReturnDate = ?
            WHERE symbol = ?
        ''', (row['Yield'], row['Price'], row['PriceDate'], row['YTDPriceReturn'], datetime.now().strftime('%Y-%m-%d'), row['Ticker']))

    conn.commit()
    conn.close()
    print("ETF YTD returns updated successfully.")
    return df


def update_security_set_ytd_returns(database_path):
    """
    Update the YTD returns for security sets in the database.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Calculate the return of each security set
    cursor.execute('SELECT id, name FROM security_sets')
    security_sets = cursor.fetchall()
    security_set_df = pd.DataFrame(columns=['SecuritySet', 'YTDPriceReturn'])

    for security_set_id, name in security_sets:
        cursor.execute('SELECT etf_id, weight FROM security_sets_etfs WHERE security_set_id = ?', (security_set_id,))
        etfs = cursor.fetchall()

        total_return = 0
        for etf_id, weight in etfs:
            cursor.execute('SELECT YTDPriceReturn FROM etfs WHERE id = ?', (etf_id,))
            ytd_price_return = cursor.fetchone()[0]
            if ytd_price_return is not None:
                total_return += weight * ytd_price_return

        # Append the security set performance to the DataFrame
        security_set_df = pd.concat([security_set_df, pd.DataFrame({
            'SecuritySet': [name],
            'YTDPriceReturn': [round(total_return, 2)]
        })], ignore_index=True)

        # Update the security set's YTDPriceReturn in the database and also update the YTDPriceReturnDate
        cursor.execute('''
            UPDATE security_sets
            SET YTDPriceReturn = ?
            , YTDPriceReturnDate = ?
            WHERE id = ?
        ''', (total_return, datetime.now().strftime('%Y-%m-%d'), security_set_id))
        print(f"Security Set: {name}, YTD Price Return: {total_return}")

    conn.commit()
    conn.close()
    print("Security set YTD returns updated successfully.")
    return security_set_df


def update_model_ytd_returns(database_path):
    """
    Update the YTD returns for models in the database.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Calculate the returns for each model
    cursor.execute('SELECT id, name FROM models')
    models = cursor.fetchall()
    model_df = pd.DataFrame(columns=['Model', 'YTDPriceReturn'])

    for model_id, name in models:
        cursor.execute('SELECT security_set_id, weight FROM model_security_set WHERE model_id = ?', (model_id,))
        security_sets = cursor.fetchall()

        total_return = 0
        for security_set_id, weight in security_sets:
            cursor.execute('SELECT YTDPriceReturn FROM security_sets WHERE id = ?', (security_set_id,))
            ytd_price_return = cursor.fetchone()[0]
            if ytd_price_return is not None:
                total_return += weight * ytd_price_return

        # Append the model performance to the DataFrame
        model_df = pd.concat([model_df, pd.DataFrame({
            'Model': [name],
            'YTDPriceReturn': [round(total_return, 2)]
        })], ignore_index=True)

        # Update the model's YTDPriceReturn in the database and also update the YTDPriceReturnDate
        cursor.execute('''
            UPDATE models
            SET YTDPriceReturn = ?
            , YTDPriceReturnDate = ?
            WHERE id = ?
        ''', (total_return, datetime.now().strftime('%Y-%m-%d'), model_id))

        print(f"Model: {name}, YTD Price Return: {total_return}")

    conn.commit()
    conn.close()
    print("Model YTD returns updated successfully.")
    return model_df
