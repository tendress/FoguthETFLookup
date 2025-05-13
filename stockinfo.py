import yfinance as yf
import pandas as pd
import sqlite3
import json


class StockInfo:
    def __init__(self, database_path):
        """Initialize the database connection"""
        self.database_path = database_path

    def get_etfs(self):
        """Fetch the list of ETFs from the etfs table"""
        conn = sqlite3.connect(self.database_path)
        query = "SELECT symbol FROM etfs"
        etfs = pd.read_sql_query(query, conn)
        conn.close()
        return etfs['symbol'].tolist()

    def fetch_and_store_etf_info(self):
        """Fetch ETF info using yfinance and update the etf_infos table"""
        etfs = self.get_etfs()
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        for ticker in etfs:
            try:
                # Fetch ETF info using yfinance
                new_ticker = yf.Ticker(ticker)
                etf_info = new_ticker.info

                # Add the ticker symbol to the data
                etf_info['symbol'] = ticker

                # Ensure all keys in the JSON match columns in the etf_infos table
                self.update_etf_infos_table(cursor, etf_info)

                print(f"Successfully updated info for {ticker}")
            except Exception as e:
                print(f"Error fetching info for {ticker}: {e}")

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    def update_etf_infos_table(self, cursor, etf_info):
        """Update the etf_infos table with the data from the JSON"""
        # Get the existing columns in the etf_infos table
        cursor.execute("PRAGMA table_info(etf_infos)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # Add any missing columns to the table
        for key in etf_info.keys():
            if key not in existing_columns:
                cursor.execute(f"ALTER TABLE etf_infos ADD COLUMN {key} TEXT")
                print(f"Added missing column: {key}")

        # Prepare the data for insertion
        columns = ", ".join(etf_info.keys())
        placeholders = ", ".join(["?"] * len(etf_info))
        values = [str(value) if isinstance(value, (list, dict)) else value for value in etf_info.values()]

        # Insert or update the row in the table
        cursor.execute(f'''
            INSERT INTO etf_infos ({columns})
            VALUES ({placeholders})
            ON CONFLICT(symbol) DO UPDATE SET
            {", ".join([f"{key} = excluded.{key}" for key in etf_info.keys()])}
        ''', values)


# Example usage
if __name__ == "__main__":
    database_path = 'foguth_etf_models.db'  # Path to your SQLite database
    stock_info = StockInfo(database_path)

    # Fetch and store ETF info in the etf_infos table
    stock_info.fetch_and_store_etf_info()