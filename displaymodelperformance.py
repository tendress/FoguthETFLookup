import sqlite3
import pandas as pd
import streamlit as st
from updateytdreturnsmodule import update_etf_ytd_returns, update_security_set_ytd_returns, update_model_ytd_returns

def display_model_performance():
    st.title("Model Performance")
    st.write("This page displays model performance metrics.")

    database_path = 'foguth_etf_models.db'

    # Load the models table
    def load_models_table():
        """
        Load the models table from the database.
        """
        try:
            conn = sqlite3.connect(database_path)
            query = """
                SELECT name AS Name, 
                       YTDPriceReturn AS YTDReturn, 
                       YTDPriceReturnDate AS AsOf, 
                       yield AS AnnualYield, 
                       ExpenseRatio 
                FROM models
            """
            models_df = pd.read_sql_query(query, conn)
            conn.close()
            return models_df
        except Exception as e:
            st.error(f"Error loading models table: {e}")
            return pd.DataFrame()

    # Load the security_sets table
    def load_security_sets_table():
        """
        Load the security_sets table from the database.
        """
        try:
            conn = sqlite3.connect(database_path)
            query = """
                SELECT name AS Name, 
                       YTDPriceReturn AS YTDReturn, 
                       YTDPriceReturnDate AS AsOf, 
                       yield AS Yield 
                FROM security_sets
            """
            security_sets_df = pd.read_sql_query(query, conn)
            conn.close()
            return security_sets_df
        except Exception as e:
            st.error(f"Error loading Security Sets table: {e}")
            return pd.DataFrame()

    # Load data from the database
    models_df = load_models_table()
    security_sets_df = load_security_sets_table()

    # Display the models table
    st.header("Year-To-Date Model Performance")
    if not models_df.empty:
        # Sort by YTDPriceReturn in descending order
        if 'YTDReturn' in models_df.columns:
            models_df = models_df.sort_values(by='YTDReturn', ascending=False).reset_index(drop=True)

        # Format the Yield and ExpenseRatio columns
        if 'ExpenseRatio' in models_df.columns:
            models_df['ExpenseRatio'] = models_df['ExpenseRatio'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

        # Display the DataFrame
        st.dataframe(models_df, use_container_width=True, height=500, hide_index=True)
    else:
        st.warning("No data available in the models table.")

    # Display the security_sets table
    st.header("Year-To-Date Security Sets Performance")
    if not security_sets_df.empty:
        # Sort by YTDPriceReturn in descending order
        if 'YTDReturn' in security_sets_df.columns:
            security_sets_df = security_sets_df.sort_values(by='YTDReturn', ascending=False).reset_index(drop=True)

        # Display the DataFrame
        st.dataframe(security_sets_df, use_container_width=True, height=500, hide_index=True)
    else:
        st.warning("No data available in the security sets table.")

    # Sidebar for benchmark YTD performance
    def get_ytd_price_return(ticker, database_path):
        """
        Calculate the YTD time-weighted return (TWR) for a given ticker using the etf_prices table.
        """
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()

            # Fetch the etf_id for the given ticker
            cursor.execute('SELECT id FROM etfs WHERE symbol = ?', (ticker,))
            etf_id = cursor.fetchone()

            if etf_id is None:
                st.error(f"Ticker {ticker} not found in the database.")
                return None

            etf_id = etf_id[0]

            # Fetch all daily close prices for the current year
            cursor.execute('''
                SELECT Date, Close
                FROM etf_prices
                WHERE etf_id = ? AND strftime('%Y', Date) = strftime('%Y', 'now')
                ORDER BY Date ASC
            ''', (etf_id,))
            price_data = cursor.fetchall()

            # Close the database connection
            conn.close()

            # Ensure there is enough data to calculate TWR
            if len(price_data) < 2:
                st.warning(f"Not enough data to calculate TWR for {ticker}.")
                return None

            # Convert the data to a DataFrame
            price_df = pd.DataFrame(price_data, columns=['Date', 'Close'])
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            price_df.set_index('Date', inplace=True)

            # Calculate daily returns
            price_df['Daily Return'] = price_df['Close'].pct_change()

            # Calculate the Time-Weighted Return (TWR)
            twr = (1 + price_df['Daily Return']).prod() - 1  # Geometric mean of returns

            # Convert TWR to a percentage
            return twr * 100

        except Exception as e:
            st.error(f"Error fetching YTD time-weighted return for {ticker}: {e}")
            return None

    # Display benchmark YTD performance in the sidebar
    st.sidebar.title("Benchmarks")
    spy_ytd_return = get_ytd_price_return('SPY', database_path)
    qqqm_ytd_return = get_ytd_price_return('QQQM', database_path)
    dia_ytd_return = get_ytd_price_return('DIA', database_path)

    st.sidebar.write(f"S&P 500 YTD: {spy_ytd_return:.2f}%" if spy_ytd_return is not None else "SPY data not available")
    st.sidebar.write(f"Nasdaq YTD: {qqqm_ytd_return:.2f}%" if qqqm_ytd_return is not None else "QQQM data not available")
    st.sidebar.write(f"Dow Jones YTD: {dia_ytd_return:.2f}%" if dia_ytd_return is not None else "DIA data not available")

    # Add a button to update the data in the database
    if st.button("Update Data"):
        st.write("Updating ETF YTD returns...")
        update_etf_ytd_returns(database_path)
        st.write("ETF YTD returns updated successfully!")

        st.write("Updating Security Set YTD returns...")
        update_security_set_ytd_returns(database_path)
        st.write("Security Set YTD returns updated successfully!")

        st.write("Updating Model YTD returns...")
        update_model_ytd_returns(database_path)
        st.write("Model YTD returns updated successfully!")

        st.success("All data updated successfully! Refresh the page to see the latest data.")
