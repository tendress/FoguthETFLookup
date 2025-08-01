import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
import datetime as dt
from fredapi import Fred
#from updateytdreturnsmodule import update_etf_ytd_returns, update_security_set_ytd_returns, update_model_ytd_returns

# --- ETF Price and Info Update Functions --- # 

## This function updates the historical daily price data for all tickers in the etfs table and inserts a new daily close price into the etf_prices table.

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
    tickers = list(ticker_to_etf_id.keys())
    try:
        all_data = yf.download(tickers, start='1994-01-01', end=enddate, interval="1d", group_by="ticker")
        if all_data.empty:
            print("No data returned for the tickers. Exiting...")
        else:
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
    print("Daily close price data has been stored in the etf_prices table.")


# --- ETF Info Update Class --- #
## This one is important because it gets fills the etf_infos table with the latest information about each ETF, including its symbol, name, and other details.


class StockInfo:
    def __init__(self, database_path):
        self.database_path = database_path

    def get_etfs(self):
        conn = sqlite3.connect(self.database_path)
        query = "SELECT symbol FROM etfs"
        etfs = pd.read_sql_query(query, conn)
        conn.close()
        return etfs['symbol'].tolist()

    def fetch_and_store_etf_info(self):
        etfs = self.get_etfs()
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        for ticker in etfs:
            try:
                new_ticker = yf.Ticker(ticker)
                etf_info = new_ticker.info
                etf_info['symbol'] = ticker
                self.update_etf_infos_table(cursor, etf_info)
                print(f"Successfully updated info for {ticker}")
            except Exception as e:
                print(f"Error fetching info for {ticker}: {e}")
        conn.commit()
        conn.close()

    def update_etf_infos_table(self, cursor, etf_info):
        cursor.execute("PRAGMA table_info(etf_infos)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        for key in etf_info.keys():
            if key not in existing_columns:
                cursor.execute(f"ALTER TABLE etf_infos ADD COLUMN {key} TEXT")
                print(f"Added missing column: {key}")
        columns = ", ".join(etf_info.keys())
        placeholders = ", ".join(["?"] * len(etf_info))
        values = [str(value) if isinstance(value, (list, dict)) else value for value in etf_info.values()]
        cursor.execute(f'''
            INSERT INTO etf_infos ({columns})
            VALUES ({placeholders})
            ON CONFLICT(symbol) DO UPDATE SET
            {", ".join([f"{key} = excluded.{key}" for key in etf_info.keys()])}
        ''', values)



## Function that updates the etf_prices.Close column with the latest daily close prices from the etf_infos table for each ETF in the etfs table.
def update_etf_prices_with_market_price(database_path):
    """
    For each ETF, update or insert today's price in etf_prices using etf_infos.regularMarketPrice.
    """
    import datetime
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Get today's date in YYYY-MM-DD format
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Fetch all ETF symbols and their regularMarketPrice from etf_infos 
    cursor.execute("SELECT symbol, regularMarketPrice FROM etf_infos WHERE regularMarketPrice IS NOT NULL")
    etf_prices = cursor.fetchall()

    for symbol, regularMarketPrice in etf_prices:
        # Get etf_id from etfs table, but exclude any symbols that have a ^ (caret) at the beginning
        if symbol.startswith('^'):
            continue
        cursor.execute("SELECT id FROM etfs WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        if not result:
            continue
        etf_id = result[0]

        # Try to update the price for today; if no row exists, insert it
        cursor.execute("""
            UPDATE etf_prices
            SET Close = ?
            WHERE etf_id = ? AND Date = ?
        """, (regularMarketPrice, etf_id, today))

        if cursor.rowcount == 0:
            # No row was updated, so insert a new one
            cursor.execute("""
                INSERT INTO etf_prices (etf_id, symbol, Date, Close)
                VALUES (?, ?, ?, ?)
            """, (etf_id, symbol, today, regularMarketPrice))

    conn.commit()
    conn.close()


## Update ETF YTD Returns ##

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


## Update Security Set YTD Returns ##

def update_security_set_ytd_returns(database_path, start_date="2025-01-01"):
    """
    Update the YTD returns for security sets in the database using the security_set_prices table.
    Only includes data from the specified start_date.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch all security sets
    cursor.execute('SELECT id, name FROM security_sets')
    security_sets = cursor.fetchall()
    security_set_df = pd.DataFrame(columns=['SecuritySet', 'YTDPriceReturn'])

    for security_set_id, name in security_sets:
        # Fetch the cumulative percent change for the security set from the security_set_prices table
        cursor.execute('''
            SELECT SUM(percentChange)
            FROM security_set_prices
            WHERE security_set_id = ? AND Date >= ?
        ''', (security_set_id, start_date))
        total_percent_change = cursor.fetchone()[0]

        # If no data is found, set the return to 0
        if total_percent_change is None:
            total_percent_change = 0

        # Append the security set performance to the DataFrame
        security_set_df = pd.concat([security_set_df, pd.DataFrame({
            'SecuritySet': [name],
            'YTDPriceReturn': [round(total_percent_change, 2)]
        })], ignore_index=True)

        # Update the security set's YTDPriceReturn in the database
        cursor.execute('''
            UPDATE security_sets
            SET YTDPriceReturn = ?, YTDPriceReturnDate = ?
            WHERE id = ?
        ''', (total_percent_change, datetime.now().strftime('%Y-%m-%d'), security_set_id))

        print(f"Security Set: {name}, YTD Price Return: {total_percent_change}")

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
        # Fetch the security sets and their weights for the current model
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


def calculate_security_set_prices(database_path):
    """
    Calculate the weighted daily price of each security set based on the ETFs it contains
    and calculate the percent change from day to day.
    If there is no value for today's date, use the previous value in the table.
    """
    import pandas as pd
    from datetime import datetime

    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)

    # Fetch security sets, ETFs, and their weights
    query = """
        SELECT 
            sse.security_set_id,
            sse.etf_id,
            sse.weight,
            sse.startDate,
            sse.endDate,
            ep.Date,
            ep.Close AS etf_price
        FROM security_sets_etfs sse
        JOIN etf_prices ep ON sse.etf_id = ep.etf_id
        WHERE ep.Date BETWEEN sse.startDate AND COALESCE(sse.endDate, '9999-12-31')
        ORDER BY sse.security_set_id, ep.Date
    """
    data = pd.read_sql_query(query, conn)
    conn.close()

    if data.empty:
        print("No data found for the given query.")
        return

    # Convert Date to datetime for easier manipulation
    data['Date'] = pd.to_datetime(data['Date'])

    # Calculate the weighted price for each ETF in the security set
    data['weighted_price'] = data['weight'] * data['etf_price']

    # Group by security_set_id and Date to calculate the total weighted price for each security set
    security_set_prices = (
        data.groupby(['security_set_id', 'Date'])['weighted_price']
        .sum()
        .reset_index()
        .rename(columns={'weighted_price': 'security_set_price'})
    )

    # Ensure we have a row for today's date for each security_set_id
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    all_sets = security_set_prices['security_set_id'].unique()
    result_rows = []

    for set_id in all_sets:
        set_prices = security_set_prices[security_set_prices['security_set_id'] == set_id].copy()
        set_prices = set_prices.sort_values('Date')
        # If today's date is missing, forward fill
        if today not in set_prices['Date'].values:
            prev_row = set_prices.iloc[-1]
            new_row = prev_row.copy()
            new_row['Date'] = today
            result_rows.append(new_row)
            print(f"[Info] No value for {today.date()} in security_set_id {set_id}. Using previous value from {prev_row['Date'].date()}.")
        result_rows.append(set_prices)

    # Combine all sets back together
    security_set_prices = pd.concat(result_rows, ignore_index=True)
    security_set_prices = security_set_prices.sort_values(['security_set_id', 'Date'])

    # Calculate the percent change from day to day for each security set
    security_set_prices['percentChange'] = (
        security_set_prices.groupby('security_set_id')['security_set_price']
        .pct_change() * 100
    )
   
    
    # Save the results to the security_set_prices table in the database
    conn = sqlite3.connect(database_path)
    security_set_prices.to_sql('security_set_prices', conn, if_exists='replace', index=False)
    conn.close()

    print("Security set prices and percent changes have been calculated and saved to the 'security_set_prices' table.")
    # Save the results to the security_set_prices table in the database
    conn = sqlite3.connect(database_path)
    security_set_prices.to_sql('security_set_prices', conn, if_exists='replace', index=False)

    cursor = conn.cursor()
    # fix the dates of rebalance
    cursor.execute('''
        UPDATE security_set_prices
        SET percentChange = 0
        WHERE security_set_id = 12
        AND Date = '2025-01-02 00:00:00'
        ''')

    cursor.execute('''
        UPDATE security_set_prices
        SET percentChange = 0
        WHERE security_set_id = 6
        AND Date = '2025-05-02 00:00:00'
        ''')

    conn.commit()
    conn.close()



def update_yields_models_and_security_sets(database_path):
    """Using the etf_infos table, update the yields for each security
    set and model in the database."""
    
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch all security sets
    cursor.execute('SELECT id, name FROM security_sets')
    security_sets = cursor.fetchall()

    for security_set_id, name in security_sets:
        # Fetch the ETFs in the security set
        cursor.execute('''
            SELECT etf_id, weight FROM security_sets_etfs WHERE security_set_id = ? AND endDate IS NULL
        ''', (security_set_id,))
        etfs = cursor.fetchall()

        total_yield = 0
        total_weight = 0

        for etf_id, weight in etfs:
            cursor.execute('SELECT dividendYield FROM etf_infos WHERE symbol = (SELECT symbol FROM etfs WHERE id = ?)', (etf_id,))
            yield_value = cursor.fetchone()
            if yield_value and yield_value[0] is not None:
                total_yield += float(yield_value[0]) * weight
                total_weight += weight

        if total_weight > 0:
            average_yield = total_yield / total_weight
        else:
            average_yield = 0

        # Update the security set's yield in the database
        cursor.execute('''
            UPDATE security_sets
            SET yield = ?
            WHERE id = ?
        ''', (average_yield, security_set_id))

        print(f"Security Set: {name}, Yield: {average_yield}")

    # Fetch all models
    cursor.execute('SELECT id, name FROM models')
    models = cursor.fetchall()

    for model_id, name in models:
        # Fetch the security sets and their weights for the current model
        cursor.execute('SELECT security_set_id, weight FROM model_security_set WHERE model_id = ?', (model_id,))
        security_sets = cursor.fetchall()

        total_yield = 0
        total_weight = 0

        for security_set_id, weight in security_sets:
            cursor.execute('SELECT yield FROM security_sets WHERE id = ?', (security_set_id,))
            yield_value = cursor.fetchone()
            if yield_value and yield_value[0] is not None:
                total_yield += yield_value[0] * weight
                total_weight += weight

        if total_weight > 0:
            average_yield = total_yield / total_weight
        else:
            average_yield = 0

        # Update the model's yield in the database
        cursor.execute('''
            UPDATE models
            SET yield = ?
            WHERE id = ?
        ''', (average_yield, model_id))
        print(f"Model: {name}, Yield: {average_yield}")
    conn.commit()
    
    conn.close()
    
# --- Web Log Update Function --- #    
def update_web_log(database_path):
    """
    Update web log with current date and time.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    val1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    val2 = 'Update'
    
    val3 = 'All data for ETP site updated successfully.'
    cursor.execute('''
        INSERT INTO ffgwebUpdateLog (updateDateTime, updateType, updateDetails)
        VALUES (?, ?, ?)
    ''', (val1, val2, val3))
    conn.commit()
    conn.close()
    
    print("Web log updated successfully with current date and time.") 
# --- Main Execution ---

if __name__ == "__main__":
    database_path = 'foguth_etf_models.db'
    fred_api_key = '43370c0e912250381f6728328dfff294'
    start_date = "2025-01-01"

    # Update ETF historical prices
    # Comment this out normally, if you miss a day or two, turn this back on for one run
    update_historical_prices(database_path)

    # Update ETF info table
    stock_info = StockInfo(database_path)
    stock_info.fetch_and_store_etf_info()
    
    # Update yields for security sets and models
    update_yields_models_and_security_sets(database_path)
    
    # Update ETF prices with latest daily close prices
    update_etf_prices_with_market_price(database_path)
    
    # Update ETF YTD returns
    print("Updating ETF YTD returns...")
    update_etf_ytd_returns(database_path)

    # Calculate security set prices
    print("Calculating security set prices...")
    calculate_security_set_prices(database_path)
    
    # Update security set YTD returns
    print("Updating security set YTD returns...")
    security_set_df = update_security_set_ytd_returns(database_path, start_date)

    # Update model YTD returns
    print("Updating model YTD returns...")
    model_df = update_model_ytd_returns(database_path)

    print("All updates completed successfully.")
    
    # Update the web log
    print("Updating web log...")
    update_web_log(database_path)
    print("Web log updated successfully.")