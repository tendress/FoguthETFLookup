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

    def calculate_model_daily_returns(model_name):
        """
        Calculate daily price returns for a model using the security_set_prices table and model_security_set weights.
        """
        conn = sqlite3.connect(database_path)
        # Get security set IDs and their weights for the model
        query = '''
            SELECT 
                ss.id AS security_set_id,
                ms.weight AS model_weight
            FROM models m
            JOIN model_security_set ms ON m.id = ms.model_id
            JOIN security_sets ss ON ms.security_set_id = ss.id
            WHERE m.name = ?
        '''
        cursor = conn.cursor()
        cursor.execute(query, (model_name,))
        security_sets = cursor.fetchall()
        conn.close()

        if not security_sets:
            return pd.Series(dtype=float)

        # Create DataFrame for security set weights
        ss_weights_df = pd.DataFrame(security_sets, columns=['security_set_id', 'model_weight'])

        # Fetch daily percent changes for each security set from security_set_prices
        conn = sqlite3.connect(database_path)
        ss_data = {}
        for ss_id in ss_weights_df['security_set_id']:
            query = '''
                SELECT Date, percentChange
                FROM security_set_prices
                WHERE security_set_id = ?
                ORDER BY Date
            '''
            df = pd.read_sql_query(query, conn, params=(ss_id,))
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                # Convert percentChange to decimal returns
                ss_data[ss_id] = df['percentChange'] / 100.0
        conn.close()

        if not ss_data:
            return pd.Series(dtype=float)

        # Combine all security set returns into a DataFrame
        ss_returns = pd.DataFrame(ss_data)

        # Set weights index for multiplication
        ss_weights_df.set_index('security_set_id', inplace=True)

        # Calculate weighted daily returns for the model
        weighted_returns = ss_returns.mul(ss_weights_df['model_weight'], axis=1).sum(axis=1)

        return weighted_returns

    # Fetch models
    models_df = fetch_models()

    # Define the model groups
    model_groups = [
        ['Conservative Growth', 'Balanced Growth', 'Bullish Growth', 'Velocity', 'Opportunistic'],
        ['Conservative Value', 'Balanced Value', 'Bullish Value', 'Velocity', 'Opportunistic'],
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

    # --- Overlay selection ---
    # Fetch available symbols and their names from the database
    conn = sqlite3.connect(database_path)
    query = """
    SELECT DISTINCT symbol, name
    FROM economic_indicators
    WHERE name IS NOT NULL
    UNION
    SELECT symbol, name
    FROM etfs
    WHERE name IS NOT NULL
    """
    symbols_df = pd.read_sql_query(query, conn)
    conn.close()

    # Create a dictionary mapping symbols to their names for display
    symbol_name_mapping = dict(zip(symbols_df['symbol'], symbols_df['name']))

    # Create a list of options with symbol in front of the name
    overlay_options = ["None"] + [f"{symbol} - {name}" for symbol, name in symbol_name_mapping.items()]

    # Sidebar for selecting a single overlay
    st.sidebar.header("Overlay ETFs or Economic Indicators")
    overlay_option = st.sidebar.selectbox(
        "Select a single Economic Indicator or ETF to Overlay",
        options=overlay_options,
        index=0,
        key="overlay_selectbox"
    )
    overlay_symbol = None
    overlay_name = None
    if overlay_option != "None":
        overlay_symbol = overlay_option.split(" - ")[0]
        overlay_name = overlay_option.split(" - ", 1)[1]

    # Automatically update the graph when overlay_option changes (no button needed)
    # Remove the "Run Model Graphs" button and always show the graph

    # Calculate daily returns for each model and store them in a dictionary
    daily_returns_dict = {}
    for model_name in models_df['name']:
        daily_returns_dict[model_name] = calculate_model_daily_returns(model_name)

    # Convert the dictionary to a DataFrame
    daily_returns_df = pd.DataFrame(daily_returns_dict)

    # Filter to only include dates from 2025-01-01 onward
    daily_returns_df = daily_returns_df[daily_returns_df.index >= pd.to_datetime("2025-01-01")]

    # Create an interactive Plotly graph
    fig = go.Figure()
    # Plot selected models
    for model_name in selected_models:
        if model_name in daily_returns_df.columns:
            model_data = daily_returns_df[model_name]
            # Calculate cumulative returns as a percentage
            cumulative_returns_pct = np.cumsum(model_data) * 100
            fig.add_trace(go.Scatter(
                x=model_data.index,
                y=cumulative_returns_pct,
                mode='lines',
                name=f"{model_name} ({cumulative_returns_pct.iloc[-1]:.2f}%)"
            ))

    # Overlay the selected ETF or economic indicator
    if overlay_symbol:
        conn = sqlite3.connect(database_path)
        query = f"""
        SELECT Date, economic_value AS Close
        FROM economic_indicators
        WHERE symbol = '{overlay_symbol}'
        UNION
        SELECT Date, Close
        FROM etf_prices
        WHERE symbol = '{overlay_symbol}'
        """
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] >= pd.to_datetime("2025-01-01")]
            df = df.sort_values('Date')
            # Normalize to start at 0% for overlay (percent change from first value)
            df = df.set_index('Date')
            df = df[~df.index.duplicated(keep='first')]
            if len(df) > 0:
                norm = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=norm.index,
                    y=norm.values,
                    mode='lines',
                    name=f"{overlay_symbol} - {overlay_name} (Overlay, % Chg)"
                ))
        conn.close()

    # Customize the layout
    fig.update_layout(
        title=f"{selected_group} - Cumulative YTD Returns (with Overlay)",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns / Overlay (%)",
        legend_title="Models & Overlays",
        template="plotly_white",
        yaxis_tickformat=".2f"
    )

    # Display the interactive Plotly graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)