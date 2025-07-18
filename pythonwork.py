import sqlite3

# This will create a new database file called 'foguth_etf_models.db'
conn = sqlite3.connect('foguth_fred_indicators.db')
conn.close()

print("New database file created successfully!")