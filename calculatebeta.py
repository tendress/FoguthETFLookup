import sqlite3
import pandas as pd
import numpy as np
import streamlit as st

def calculate_beta_page():
    st.title("Beta Calculator")
    st.write("This page calculates the weighted beta for models.")

    # Function to fetch historical price data from the database
    def fetch_historical_prices(symbol, start_date, end_date, database_path):
        """
        Fetch historical price data for a given symbol from the etf_prices table.

        Args:
            symbol (str): The ticker symbol of the ETF.
            start_date (str): The start date for the time period (format: 'YYYY-MM-DD').
            end_date (str): The end date for the time period (format: 'YYYY-MM-DD').
            database_path (str): Path to the SQLite database.

        Returns:
            pd.Series: A pandas Series containing the historical close prices.
        """
        conn = sqlite3.connect(database_path)
        query = '''
            SELECT Date, Close
            FROM etf_prices
            JOIN etfs ON etf_prices.etf_id = etfs.id
            WHERE etfs.symbol = ? AND Date BETWEEN ? AND ?
            ORDER BY Date
        '''
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()

        if df.empty:
            return pd.Series(dtype=float)  # Return an empty Series if no data is found

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df['Close']

    # Function to calculate beta
    def calculate_beta(symbol, benchmark_symbol, start_date, end_date, database_path):
        """
        Calculate the beta of a stock or ETF relative to a benchmark index.

        Args:
            symbol (str): The ticker symbol of the stock or ETF.
            benchmark_symbol (str): The ticker symbol of the benchmark index.
            start_date (str): The start date for the time period (format: 'YYYY-MM-DD').
            end_date (str): The end date for the time period (format: 'YYYY-MM-DD').
            database_path (str): Path to the SQLite database.

        Returns:
            float: The beta value.
        """
        # Fetch historical price data for the asset and the benchmark from the database
        asset_data = fetch_historical_prices(symbol, start_date, end_date, database_path)
        benchmark_data = fetch_historical_prices(benchmark_symbol, start_date, end_date, database_path)

        # Ensure both datasets have the same dates
        data = pd.DataFrame({'Asset': asset_data, 'Benchmark': benchmark_data}).dropna()

        # Check if data is sufficient for calculation
        if data.empty or len(data) < 2:
            print(f"Insufficient data for {symbol} or {benchmark_symbol}.")
            return np.nan

        # Calculate daily returns
        data['Asset Returns'] = data['Asset'].pct_change()
        data['Benchmark Returns'] = data['Benchmark'].pct_change()

        # Drop NaN values after calculating returns
        data = data.dropna()

        # Calculate covariance and variance
        covariance = np.cov(data['Asset Returns'], data['Benchmark Returns'])[0, 1]
        variance = np.var(data['Benchmark Returns'])

        # Calculate beta
        beta = covariance / variance

        return beta

    # Streamlit app
    st.title("Model Weighted Beta Calculator")

    # Connect to the SQLite database
    database_path = 'foguth_etf_models.db'
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch model names
    cursor.execute('SELECT name FROM models')
    models = [row[0] for row in cursor.fetchall()]

    # Streamlit selectbox for model selection
    selected_model = st.selectbox("Select a Model", models)

    # Streamlit dropdown for benchmark selection
    # Streamlit dropdown for benchmark selection
    benchmark_symbol = st.selectbox(
        "Select a Benchmark",
        options=["S&P 500", "Dow Jones Industrial Average", "Nasdaq 100"],  # Renamed benchmark options
        index=0  # Default to S&P 500
    )

    # Map the user-friendly names back to their ticker symbols
    benchmark_mapping = {
        "S&P 500": "SPY",
        "Dow Jones Industrial Average": "DIA",
        "Nasdaq 100": "QQQM"
    }
    benchmark_ticker = benchmark_mapping[benchmark_symbol]

    # Streamlit inputs for time frame
    start_date = st.date_input(
        "Start Date",
        value=pd.Timestamp("2024-01-01"),
        min_value=pd.Timestamp("2022-01-01"),  # Limit to dates after January 1, 2022
        max_value=pd.Timestamp.now()  # Limit to today's date
    )
    end_date = st.date_input(
        "End Date",
        value=pd.Timestamp.now(),
        max_value=pd.Timestamp.now()  # Limit to today's date
    )

    if st.button("Calculate Weighted Beta"):
        # Query to get security sets and weights for the selected model
        cursor.execute('''
            SELECT ss.id AS SecuritySetID, ms.weight AS ModelWeight
            FROM models m
            JOIN model_security_set ms ON m.id = ms.model_id
            JOIN security_sets ss ON ms.security_set_id = ss.id
            WHERE m.name = ?
        ''', (selected_model,))
        security_sets = cursor.fetchall()

        total_weighted_beta = 0

        # Loop through each security set
        for security_set_id, model_weight in security_sets:
            # Query to get ETFs and weights for the security set
            cursor.execute('''
                SELECT e.symbol AS ETF, se.weight AS SecuritySetWeight
                FROM security_sets_etfs se
                JOIN etfs e ON se.etf_id = e.id
                WHERE se.security_set_id = ?
            ''', (security_set_id,))
            etfs = cursor.fetchall()

            # Calculate the weighted beta for each ETF
            for etf_symbol, security_set_weight in etfs:
                beta = calculate_beta(etf_symbol, benchmark_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), database_path)
                if not np.isnan(beta):
                    total_weighted_beta += model_weight * security_set_weight * beta

        # Display the result prominently
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <h1 style="color: #4CAF50; font-size: 48px;">Weighted Beta</h1>
                <h2 style="color: #FF5722; font-size: 36px;">{total_weighted_beta:.4f}</h2>
                <p style="font-size: 18px;">Model: <strong>{selected_model}</strong></p>
                <p style="font-size: 18px;">Time Period: <strong>{start_date} to {end_date}</strong></p>
                <p style="font-size: 18px;">Benchmark: <strong>{benchmark_symbol}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Close the database connection
    conn.close()
