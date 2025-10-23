import sqlite3
import pandas as pd
import argparse
from datetime import datetime, timedelta
import logging

def update_benchmark_returns(database_path, strategy=None, start_date=None, end_date=None, verbose=False):
    """
    Update benchmark returns table by calculating daily returns from benchmark prices.
    
    Args:
        database_path (str): Path to the SQLite database
        strategy (str, optional): Specific strategy to update. If None, updates all strategies.
        start_date (str, optional): Start date in format 'YYYY-MM-DD'. If None, starts from oldest date.
        end_date (str, optional): End date in format 'YYYY-MM-DD'. If None, uses today's date.
        verbose (bool, optional): Whether to print detailed logs.
    """
    # Setup logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set end_date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logging.info(f"Updating benchmark returns in {database_path}")
    logging.info(f"Strategy: {strategy if strategy else 'All strategies'}")
    logging.info(f"Date range: {start_date if start_date else 'Earliest'} to {end_date}")
    
    # Connect to database
    conn = sqlite3.connect(database_path)
    
    try:
        # Build query conditions
        conditions = []
        params = []
        
        if strategy:
            conditions.append("strategy_benchmark = ?")
            params.append(strategy)
        
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
            
        conditions.append("date <= ?")
        params.append(end_date)
        
        where_clause = " AND ".join(conditions)
        
        # Get benchmark prices
        query = f"""
        SELECT strategy_benchmark, date, close
        FROM benchmark_prices
        WHERE {where_clause}
        ORDER BY strategy_benchmark, date
        """
        
        logging.info(f"Executing query: {query} with params {params}")
        df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            logging.warning("No benchmark price data found for the specified criteria")
            return
            
        logging.info(f"Found {len(df)} price records")
        
        # Calculate returns by strategy
        results = []
        for strategy_name in df['strategy_benchmark'].unique():
            strategy_data = df[df['strategy_benchmark'] == strategy_name].sort_values('date')
            
            # Skip if less than 2 data points (need at least 2 for returns calculation)
            if len(strategy_data) < 2:
                logging.warning(f"Insufficient data points for {strategy_name}")
                continue
                
            # Calculate daily returns: (today's price / yesterday's price) - 1
            strategy_data['prev_price'] = strategy_data['close'].shift(1)
            strategy_data = strategy_data.dropna()  # Remove first row with NaN prev_price
            strategy_data['return'] = (strategy_data['close'] / strategy_data['prev_price']) - 1
            
            # Prepare data for insertion
            returns_data = strategy_data[['strategy_benchmark', 'date', 'return']]
            results.append(returns_data)
            
        if not results:
            logging.warning("No returns could be calculated")
            return
            
        # Combine all results
        all_returns = pd.concat(results)
        logging.info(f"Calculated {len(all_returns)} daily returns")
        
        # Create or replace benchmark_returns table
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_returns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_benchmark TEXT NOT NULL,
            date TEXT NOT NULL,
            return REAL,
            UNIQUE(strategy_benchmark, date)
        )
        """)
        
        # Insert or replace returns in the table
        for _, row in all_returns.iterrows():
            cursor.execute("""
            INSERT OR REPLACE INTO benchmark_returns (strategy_benchmark, date, return)
            VALUES (?, ?, ?)
            """, (row['strategy_benchmark'], row['date'], row['return']))
            
        conn.commit()
        logging.info(f"Successfully updated benchmark returns table with {len(all_returns)} records")
        
    except Exception as e:
        logging.error(f"Error updating benchmark returns: {str(e)}")
        conn.rollback()
        raise
        
    finally:
        conn.close()
        logging.info("Database connection closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update benchmark returns table")
    parser.add_argument("--database", "-d", default="foguthbenchmarks.db", help="Path to SQLite database file")
    parser.add_argument("--strategy", "-s", help="Specific strategy to update (default: all strategies)")
    parser.add_argument("--start-date", help="Start date in format YYYY-MM-DD (default: earliest available)")
    parser.add_argument("--end-date", help="End date in format YYYY-MM-DD (default: today)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    update_benchmark_returns(
        args.database,
        strategy=args.strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        verbose=args.verbose
    )