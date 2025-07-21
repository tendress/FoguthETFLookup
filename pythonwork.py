import sqlite3
from fredapi import Fred
import pandas as pd
from datetime import datetime

# --- FRED Economic Indicators Update ---

def update_fred_economic_indicators(database_path, api_key):
    fred_client = Fred(api_key=api_key)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Get the economic_indicator_id for each symbol
    cursor.execute("SELECT id, symbol FROM economic_indicators")
    symbol_to_id = {row[1]: row[0] for row in cursor.fetchall()}
    
    for symbol, economic_indicator_id in symbol_to_id.items():
        try:
            data = fred_client.get_series(symbol, observation_start='1994-01-01', observation_end=datetime.now().strftime('%Y-%m-%d'))
            data_df = pd.DataFrame(data, columns=['value'])
            data_df.index.name = 'date'
            data_df.reset_index(inplace=True)
            data_df['date'] = data_df['date'].dt.strftime('%Y-%m-%d')
            
            # Filter out rows where value is NaN or None
            data_df = data_df.dropna(subset=['value'])
            
            if data_df.empty:
                print(f"No valid data for {symbol}. Skipping...")
                continue
            
            # Get existing dates for this economic indicator to avoid duplicates
            cursor.execute("""
                SELECT date FROM indicator_values 
                WHERE economic_indicator_id = ?
            """, (economic_indicator_id,))
            existing_dates = set(row[0] for row in cursor.fetchall())
            
            # Filter out dates that already exist in the database
            new_data = data_df[~data_df['date'].isin(existing_dates)]
            
            if new_data.empty:
                print(f"No new data for {symbol}. All dates already exist in database.")
                continue
            
            rows_inserted = 0
            for _, row in new_data.iterrows():
                # Double-check that value is not None/NaN before inserting
                if pd.notna(row['value']) and row['value'] is not None:
                    cursor.execute("""
                        INSERT INTO indicator_values (economic_indicator_id, value, date)
                        VALUES (?, ?, ?)
                    """, (economic_indicator_id, row['value'], row['date']))
                    rows_inserted += 1
            
            print(f"Data for {symbol}: {rows_inserted} new rows inserted successfully.")
            
        except Exception as e:
            print(f"Error fetching or inserting data for {symbol}: {e}")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    database_path = 'foguth_fred_indicators.db'
    fred_api_key = '43370c0e912250381f6728328dfff294'
    
    update_fred_economic_indicators(database_path, fred_api_key)