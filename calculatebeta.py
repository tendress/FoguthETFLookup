import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import streamlit as st

# Function to calculate beta
def calculate_beta(symbol, benchmark_symbol, start_date, end_date):
    """
    Calculate the beta of a stock or ETF relative to a benchmark index.

    Args:
        symbol (str): The ticker symbol of the stock or ETF.
        benchmark_symbol (str): The ticker symbol of the benchmark index (e.g., '^GSPC' for S&P 500).
        start_date (str): The start date for the time period (format: 'YYYY-MM-DD').
        end_date (str): The end date for the time period (format: 'YYYY-MM-DD').

    Returns:
        float: The beta value.
    """
    # Fetch historical price data for the asset and the benchmark
    asset_data = yf.Ticker(symbol).history(start=start_date, end=end_date)['Close']
    benchmark_data = yf.Ticker(benchmark_symbol).history(start=start_date, end=end_date)['Close']

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

# Streamlit inputs for time frame
start_date = st.date_input("Start Date", value=pd.Timestamp("2024-01-01"))
end_date = st.date_input("End Date", value=pd.Timestamp.now())

# Benchmark symbol
benchmark_symbol = '^GSPC'  # S&P 500 as the benchmark

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
            beta = calculate_beta(etf_symbol, benchmark_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
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
        </div>
        """,
        unsafe_allow_html=True
    )
# Close the database connection
conn.close()
