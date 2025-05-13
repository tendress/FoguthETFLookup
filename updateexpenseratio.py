import sqlite3
import yfinance as yf

def update_etf_expense_ratios(database_path):
    """
    Fetch the expense ratio for each ETF in the etfs table and update the table.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the list of symbols from the etfs table
    cursor.execute('SELECT symbol FROM etfs')
    etf_symbols = [row[0] for row in cursor.fetchall()]
    print("Fetched ETF symbols:", etf_symbols)

    # Iterate through each symbol and fetch the expense ratio
    for symbol in etf_symbols:
        try:
            # Fetch ETF data using yfinance
            etf = yf.Ticker(symbol)
            expense_ratio = etf.info.get('netExpenseRatio')  # Get the expense ratio
            
            if expense_ratio is not None:  # If expense ratio is found, update the table
                cursor.execute('UPDATE etfs SET expense_ratio = ? WHERE symbol = ?', (expense_ratio, symbol))
                print(f"Updated expense ratio for {symbol}: {expense_ratio}")
            else:
                print(f"No expense ratio found for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Expense ratios updated successfully.")

# Path to the SQLite database
database_path = 'foguth_etf_models.db'

# Run the script
update_etf_expense_ratios(database_path)