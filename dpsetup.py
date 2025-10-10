# create a new database called foguthbenchmarks.db
import sqlite3

conn = sqlite3.connect('foguthbenchmarks.db')
cursor = conn.cursor()

try:
    conn.commit()

except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
    
finally:
    conn.close()
import sqlite3
import os

def copy_tables(source_db_path, dest_db_path, tables_to_copy):
    """
    Copy specified tables from source database to destination database
    """
    print(f"Copying tables from {source_db_path} to {dest_db_path}")
    
    # Connect to both databases
    source_conn = sqlite3.connect(source_db_path)
    dest_conn = sqlite3.connect(dest_db_path)
    
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()
    
    try:
        # Process each table
        for table_name in tables_to_copy:
            print(f"Processing table: {table_name}")
            
            # 1. Get table schema from source
            source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            create_table_sql = source_cursor.fetchone()
            
            if not create_table_sql:
                print(f"Table {table_name} not found in source database")
                continue
                
            # 2. Create table in destination
            dest_cursor.execute("DROP TABLE IF EXISTS " + table_name)
            dest_cursor.execute(create_table_sql[0])
            
            # 3. Get all data from source
            source_cursor.execute(f"SELECT * FROM {table_name}")
            rows = source_cursor.fetchall()
            
            if not rows:
                print(f"No data found in {table_name}")
                continue
                
            # 4. Get column count for parameterized query
            column_count = len(rows[0])
            placeholders = ','.join(['?' for _ in range(column_count)])
            
            # 5. Insert data into destination
            dest_cursor.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                rows
            )
            
            print(f"Copied {len(rows)} rows from {table_name}")
            
        dest_conn.commit()
        print("All tables copied successfully")
        
    except Exception as e:
        print(f"Error during copy: {e}")
        dest_conn.rollback()
        
    finally:
        source_conn.close()
        dest_conn.close()

if __name__ == "__main__":
    source_db = 'foguth_etf_models.db'
    dest_db = 'foguthbenchmarks.db'
    tables = ['benchmarks', 'benchmark_returns', 'benchmark_prices']
    
    # Verify source database exists
    if not os.path.exists(source_db):
        print(f"Error: Source database {source_db} not found")
    else:
        copy_tables(source_db, dest_db, tables)