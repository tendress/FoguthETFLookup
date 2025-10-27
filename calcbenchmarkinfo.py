import sqlite3
import pandas as pd
import datetime as dt

def calculate_weighted_benchmarks(models_db_path, bench_db_path):
    """
    Read ETF prices from models DB (models_db_path) and
    create/store weighted benchmark prices & daily returns in bench DB (bench_db_path).
    """
    print(f"Starting benchmark calculations.\nModels DB: {models_db_path}\nBench DB:  {bench_db_path}")

    conn_models = sqlite3.connect(models_db_path)
    conn_bench = sqlite3.connect(bench_db_path)
    c_models = conn_models.cursor()
    c_bench = conn_bench.cursor()

    try:
        # Create target tables in benchmark DB if they don't exist (use close_price consistently)
        c_bench.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_benchmark TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                UNIQUE (strategy_benchmark, date)
            )
        """)
        c_bench.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_returns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_benchmark TEXT NOT NULL,
                date TEXT NOT NULL,
                return REAL,
                UNIQUE (strategy_benchmark, date)
            )
        """)
        conn_bench.commit()

        # Read distinct strategies from benchmarks table in bench DB
        c_bench.execute("SELECT DISTINCT strategy FROM benchmarks")
        strategies = [row[0] for row in c_bench.fetchall()]

        if not strategies:
            print("No strategies found in benchmarks table.")
            return

        print(f"Found {len(strategies)} strategies")

        for strategy in strategies:
            print(f"\nProcessing strategy: {strategy}")

            # Get component ETFs (symbols) and weights from benchmarks table in bench DB
            c_bench.execute("""
                SELECT benchmark_etf, weight FROM benchmarks
                WHERE strategy = ?
            """, (strategy,))
            components = c_bench.fetchall()  # list of (symbol, weight)

            if not components:
                print(f"No components for strategy {strategy}, skipping.")
                continue

            symbols = [row[0] for row in components]

            # Get all available dates across those symbols from models DB
            placeholders = ','.join(['?'] * len(symbols))
            c_models.execute(f"""
                SELECT DISTINCT Date FROM etf_prices
                WHERE symbol IN ({placeholders})
                ORDER BY Date
            """, symbols)
            dates = [r[0] for r in c_models.fetchall()]

            if not dates:
                print(f"No price dates found for components of {strategy}, skipping.")
                continue

            weighted_prices = []

            for date in dates:
                total_weighted_price = 0.0
                total_weight = 0.0

                for symbol, weight in components:
                    c_models.execute("""
                        SELECT Close FROM etf_prices
                        WHERE symbol = ? AND Date = ?
                    """, (symbol, date))
                    res = c_models.fetchone()
                    if res and res[0] is not None:
                        price = res[0]
                        total_weighted_price += price * weight
                        total_weight += weight

                # Only store if at least one component contributed
                if total_weight > 0:
                    # normalize if weights don't sum to 1 (defensive)
                    if abs(total_weight - 1.0) > 0.001:
                        total_weighted_price = total_weighted_price / total_weight

                    # Insert or replace into benchmark_prices in bench DB
                    c_bench.execute("""
                        INSERT OR REPLACE INTO benchmark_prices (strategy_benchmark, date, close)
                        VALUES (?, ?, ?)
                    """, (strategy, date, total_weighted_price))
                    weighted_prices.append((date, total_weighted_price))

            conn_bench.commit()
            print(f"Inserted/updated {len(weighted_prices)} benchmark price rows for {strategy}")

            # Calculate daily returns from weighted_prices (sorted by date)
            if len(weighted_prices) >= 2:
                # ensure sorted by date
                weighted_prices.sort(key=lambda x: x[0])
                for i in range(1, len(weighted_prices)):
                    prev_date, prev_price = weighted_prices[i-1]
                    curr_date, curr_price = weighted_prices[i]

                    # avoid division by zero
                    if prev_price and prev_price != 0:
                        daily_return = (curr_price / prev_price) - 1.0
                        # store return in benchmark_returns.close_price
                        c_bench.execute("""
                            INSERT OR REPLACE INTO benchmark_returns (strategy_benchmark, date, return)
                            VALUES (?, ?, ?)
                        """, (strategy, curr_date, daily_return))
                conn_bench.commit()
                print(f"Calculated and stored {len(weighted_prices)-1} returns for {strategy}")
            else:
                print(f"Not enough price points to calculate returns for {strategy}")

        print("\nSuccessfully calculated all weighted benchmark prices and returns.")

    except Exception as e:
        print("Error during calculation:", e)
        conn_bench.rollback()

    finally:
        conn_models.close()
        conn_bench.close()


if __name__ == "__main__":
    models_db = "foguth_etf_models.db"   # source of `etf_prices`
    bench_db = "foguthbenchmarks.db"     # destination for `benchmarks`, `benchmark_prices`, `benchmark_returns`
    calculate_weighted_benchmarks(models_db, bench_db)