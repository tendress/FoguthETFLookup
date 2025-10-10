import sqlite3
import pandas as pd
import datetime as dt

def calculate_weighted_benchmarks(database_path):
    """
    Calculate daily weighted benchmark prices and returns for each strategy.
    """
    print(f"Starting benchmark calculations using {database_path}")
    
    # Connect to database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    try:
        # Create benchmark_prices table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_benchmark TEXT NOT NULL, 
                date TEXT NOT NULL,
                close_price REAL,
                UNIQUE (strategy_benchmark, date)
            )
        """)
        conn.commit()
        
        # Get distinct strategies from benchmarks table
        cursor.execute("SELECT DISTINCT strategy FROM benchmarks")
        strategies = [row[0] for row in cursor.fetchall()]
        
        if not strategies:
            print("No strategies found in benchmarks table.")
            return
        
        print(f"Found {len(strategies)} strategies: {strategies}")
        
        for strategy in strategies:
            print(f"Processing strategy: {strategy}")
            
            # Get ETFs and weights for this strategy
            cursor.execute("""
                SELECT benchmark_etf, weight FROM benchmarks
                WHERE strategy = ?
            """, (strategy,))
            strategy_etfs = cursor.fetchall()
            
            if not strategy_etfs:
                print(f"No ETFs found for strategy: {strategy}")
                continue
            
            # Get all dates for which we have price data
            etf_ids = [etf_id for etf_id, _ in strategy_etfs]
            placeholders = ','.join(['?' for _ in etf_ids])
            
            cursor.execute(f"""
                SELECT DISTINCT Date FROM etf_prices
                WHERE symbol IN ({placeholders})
                ORDER BY Date
            """, etf_ids)
            dates = [row[0] for row in cursor.fetchall()]
            
            # Calculate weighted price for each date
            weighted_prices = []
            for date in dates:
                total_weighted_price = 0
                total_weight = 0
                
                for etf_id, weight in strategy_etfs:
                    cursor.execute("""
                        SELECT Close FROM etf_prices
                        WHERE symbol = ? AND Date = ?
                    """, (etf_id, date))
                    
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        price = result[0]
                        total_weighted_price += price * weight
                        total_weight += weight
                
                # Only record if we have data
                if total_weight > 0:
                    # Normalize if weights don't sum to 1
                    if abs(total_weight - 1.0) > 0.001:
                        total_weighted_price /= total_weight
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO benchmark_prices
                        (strategy_benchmark, date, close)
                        VALUES (?, ?, ?)
                    """, (strategy, date, total_weighted_price))
                    
                    weighted_prices.append((date, total_weighted_price))
            
            # Calculate daily returns
            for i in range(1, len(weighted_prices)):
                prev_date, prev_price = weighted_prices[i-1]
                curr_date, curr_price = weighted_prices[i]
                
                if prev_price > 0:
                    daily_return = (curr_price / prev_price) - 1
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO benchmark_returns
                        (strategy_benchmark, date, return)
                        VALUES (?, ?, ?)
                    """, (strategy, curr_date, daily_return))
            
            print(f"Processed {len(weighted_prices)} prices and {len(weighted_prices)-1} returns for {strategy}")
            conn.commit()
        
        print("Successfully calculated all weighted benchmark returns.")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        
    finally:
        conn.close()

if __name__ == "__main__":
    database_path = "foguth_etf_models.db"
    calculate_weighted_benchmarks(database_path)