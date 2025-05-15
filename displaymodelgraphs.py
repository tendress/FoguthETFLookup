import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

def display_model_graphs():
    st.title("Model Graphs")
    st.write("This page displays interactive graphs for model performance.")

    # Database connection
    database_path = 'foguth_etf_models.db'

    # Remove @st.cache_data
    def fetch_models():
        """
        Fetch model names and YTD returns from the database.
        """
        conn = sqlite3.connect(database_path)
        query = '''
            SELECT name, YTDPriceReturn
            FROM models
        '''
        models = pd.read_sql_query(query, conn)
        conn.close()
        return models

    # Remove @st.cache_data
    def calculate_model_daily_returns(model_name):
        """
        Calculate daily price returns for a model using YTD data from the etf_prices table.
        """
        conn = sqlite3.connect(database_path)
        query = '''
            SELECT 
                e.symbol AS ETF,
                ms.weight * se.weight AS Weight
            FROM models m
            JOIN model_security_set ms ON m.id = ms.model_id
            JOIN security_sets_etfs se ON ms.security_set_id = se.security_set_id
            JOIN etfs e ON se.etf_id = e.id
            WHERE m.name = ?
        '''
        cursor = conn.cursor()
        cursor.execute(query, (model_name,))
        etf_weights = cursor.fetchall()
        conn.close()

        # Create a DataFrame for ETF weights
        etf_weights_df = pd.DataFrame(etf_weights, columns=['ETF', 'Weight'])

        # Fetch historical price data for the ETFs (YTD) from the database
        etf_data = {}
        start_date = dt.datetime(dt.datetime.now().year, 1, 1).strftime('%Y-%m-%d')  # Start from January 1st of the current year
        conn = sqlite3.connect(database_path)
        for etf in etf_weights_df['ETF']:
            query = '''
                SELECT Date, Close
                FROM etf_prices
                JOIN etfs ON etf_prices.etf_id = etfs.id
                WHERE etfs.symbol = ? AND Date >= ?
                ORDER BY Date
            '''
            df = pd.read_sql_query(query, conn, params=(etf, start_date))
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                etf_data[etf] = df['Close']
        conn.close()

        # Combine ETF price data into a single DataFrame
        etf_prices = pd.DataFrame(etf_data)

        # Calculate daily returns for each ETF
        etf_daily_returns = etf_prices.pct_change(fill_method=None)

        # Calculate the weighted daily returns for the model
        etf_weights_df.set_index('ETF', inplace=True)
        weighted_returns = etf_daily_returns.mul(etf_weights_df['Weight'], axis=1).sum(axis=1)

        return weighted_returns

    # Fetch models
    models_df = fetch_models()

    # Define the model groups
    model_groups = [
        ['Conservative Growth', 'Balanced Growth', 'Bullish Growth', 'Aggressive', 'Momentum'],
        ['Conservative Value', 'Balanced Value', 'Bullish Value', 'Aggressive', 'Momentum'],
        ['Rising Dividend Conservative', 'Rising Dividend Balanced', 'Rising Dividend Bullish', 'Rising Dividend Aggressive', 'Rising Dividend Momentum']
    ]

    # Streamlit selectbox for group selection
    group_mapping = {
        "Growth Tilt": 0,
        "Value Tilt": 1,
        "Rising Dividend": 2
    }
    selected_group = st.selectbox("Select a Model Group", list(group_mapping.keys()))
    group_index = group_mapping[selected_group]  # Map the selected group to its index
    selected_models = model_groups[group_index]

    # Display a button to initiate the calculation
    if st.button("Run Model Graphs"):
        # Calculate daily returns for each model and store them in a dictionary
        daily_returns_dict = {}
        for model_name in models_df['name']:
            daily_returns_dict[model_name] = calculate_model_daily_returns(model_name)

        # Convert the dictionary to a DataFrame
        daily_returns_df = pd.DataFrame(daily_returns_dict)

        # Create an interactive Plotly graph
        fig = go.Figure()
        for model_name in selected_models:
            if model_name in daily_returns_df.columns:
                model_data = daily_returns_df[model_name]
                fig.add_trace(go.Scatter(
                    x=model_data.index,
                    y=np.cumsum(model_data),
                    mode='lines',
                    name=model_name
                ))

        # Customize the layout
        fig.update_layout(
            title=f"{selected_group} - Cumulative YTD Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            legend_title="Models",
            template="plotly_white"
        )

        # Display the interactive Plotly graph in Streamlit
        st.plotly_chart(fig, use_container_width=True)
