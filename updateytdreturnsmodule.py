import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime

def update_etf_ytd_returns(database_path):
    """
    Update the YTD returns for ETFs in the database using the etf_prices table.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the list of tickers and their corresponding etf_id
    cursor.execute('SELECT id, symbol FROM etfs')
    etf_data = cursor.fetchall()  # Fetch all rows as a list of tuples
    print("List of tickers fetched from the etfs table:")
    print(etf_data)

    # Loop through each ETF and calculate the YTD return
    for etf_id, ticker in etf_data:
        # Fetch the first and most recent close prices for the current year
        cursor.execute('''
            SELECT Close FROM etf_prices
            WHERE etf_id = ? AND Date = (
                SELECT MIN(Date) FROM etf_prices
                WHERE etf_id = ? AND strftime('%Y', Date) = strftime('%Y', 'now')
            )
        ''', (etf_id, etf_id))
        start_price = cursor.fetchone()

        cursor.execute('''
            SELECT Close FROM etf_prices
            WHERE etf_id = ? AND Date = (
                SELECT MAX(Date) FROM etf_prices
                WHERE etf_id = ? AND strftime('%Y', Date) = strftime('%Y', 'now')
            )
        ''', (etf_id, etf_id))
        end_price = cursor.fetchone()

        # Calculate YTD return
        if start_price and end_price:
            start_price = start_price[0]
            end_price = end_price[0]
            ytd_price_return = ((end_price - start_price) / start_price) * 100 if start_price else None
        else:
            ytd_price_return = None

        # Update the YTD return in the database
        cursor.execute('''
            UPDATE etfs
            SET YTDPriceReturn = ?, PriceReturnDate = ?
            WHERE id = ?
        ''', (ytd_price_return, datetime.now().strftime('%Y-%m-%d'), etf_id))

        print(f"ETF: {ticker}, YTD Price Return: {ytd_price_return}")

    conn.commit()
    conn.close()
    print("ETF YTD returns updated successfully.")
    return ytd_price_return

def update_etf_yields(database_path):
    """
    Update the yields for ETFs in the database.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the list of tickers
    cursor.execute('SELECT id, symbol FROM etfs')
    etf_data = cursor.fetchall()  # Fetch all rows as a list of tuples
    print("List of tickers fetched from the etfs table:")
    print(etf_data)

    # Loop through each ETF and update the yield
    for etf_id, ticker in etf_data:
        try:
            # Fetch yield information from yfinance
            yield_info = yf.Ticker(ticker).info.get('yield', None)

            # Update the yield in the database
            cursor.execute('''
                UPDATE etfs
                SET yield = ?
                WHERE id = ?
            ''', (yield_info, etf_id))

            print(f"ETF: {ticker}, Yield: {yield_info}")
        except Exception as e:
            print(f"Error fetching yield for {ticker}: {e}")

    conn.commit()
    conn.close()
    print("ETF yields updated successfully.")

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
