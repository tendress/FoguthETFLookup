import sqlite3
import pandas as pd
import streamlit as st
import datetime  # Import datetime for date calculations
import matplotlib.pyplot as plt

def display_correlation_matrix():
    def safe_pyplot(fig):
        try:
            st.pyplot(fig)
        except TypeError:
            st.pyplot(fig)

    st.title("ETF Correlation Matrix")
    st.write("This page displays the correlation matrix for ETFs based on their historical prices.")

    database_path = 'foguth_etf_models.db'
    conn = sqlite3.connect(database_path)

    # Fetch the list of models
    models_df = pd.read_sql_query("SELECT name FROM models ORDER BY name", conn)
    models = models_df["name"].tolist()

    if not models:
        st.warning("No models found in the database.")
        conn.close()
        return

    # Let the user select a model
    selected_model = st.sidebar.selectbox("Select a Model", models, key="corr_model")

    # Fetch the ETFs associated with the selected model
    etf_query = """
        SELECT e.symbol
        FROM etfs e
        JOIN security_sets_etfs sse ON e.id = sse.etf_id
        JOIN security_sets ss ON sse.security_set_id = ss.id
        JOIN model_security_set mss ON ss.id = mss.security_set_id
        JOIN models m ON mss.model_id = m.id
        WHERE m.name = ?
          AND sse.endDate IS NULL
    """
    etf_symbols_df = pd.read_sql_query(etf_query, conn, params=(selected_model,))
    etf_symbols = sorted(etf_symbols_df["symbol"].dropna().unique().tolist())

    if not etf_symbols:
        st.warning(f"No ETFs found for the selected model: {selected_model}.")
        conn.close()
        return

    # Let the user select a date range (default to Year to Date)
    st.sidebar.header("Filter by Date Range")
    today = datetime.date.today()
    start_of_year = datetime.date(today.year, 1, 1)
    start_date = st.sidebar.date_input("Start Date", value=start_of_year, key="corr_start_date")
    end_date = st.sidebar.date_input("End Date", value=today, key="corr_end_date")

    if start_date > end_date:
        st.error("Start date must be before end date.")
        conn.close()
        return

    # Fetch price data for the ETFs from the etf_prices table
    price_query = """
        SELECT symbol, Date, Close
        FROM etf_prices
        WHERE symbol IN ({}) AND substr(Date, 1, 10) BETWEEN ? AND ?
        ORDER BY Date ASC
    """.format(','.join(['?'] * len(etf_symbols)))

    params = etf_symbols + [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
    price_data = pd.read_sql_query(price_query, conn, params=params)

    # Close the database connection
    conn.close()

    if price_data.empty:
        st.warning("No price data found for the selected ETFs and date range.")
        return

    # Pivot the data to create a DataFrame with symbols as columns and dates as the index
    price_data['Date'] = pd.to_datetime(price_data['Date'], errors='coerce')
    price_data['Close'] = pd.to_numeric(price_data['Close'], errors='coerce')
    price_data = price_data.dropna(subset=['Date', 'symbol', 'Close'])

    pivoted_data = price_data.pivot_table(
        index='Date',
        columns='symbol',
        values='Close',
        aggfunc='last'
    )

    if pivoted_data.shape[0] < 2 or pivoted_data.shape[1] < 2:
        st.warning("Not enough data points to compute a correlation matrix for this range.")
        return

    # Calculate the correlation matrix
    correlation_matrix = pivoted_data.corr()

    if correlation_matrix.empty:
        st.warning("Correlation matrix could not be computed for the selected data.")
        return

    # Render heatmap with Matplotlib (avoids Plotly/Arrow frontend issues)
    matrix_size = len(correlation_matrix.columns)
    fig_width = max(8, min(20, matrix_size * 0.7))
    fig_height = max(6, min(20, matrix_size * 0.7))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    heat = ax.imshow(correlation_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f"ETF Correlation Matrix ({selected_model})")
    ax.set_xlabel("ETFs")
    ax.set_ylabel("ETFs")

    tick_labels = correlation_matrix.columns.tolist()
    ax.set_xticks(range(matrix_size))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(matrix_size))
    ax.set_yticklabels(tick_labels, fontsize=9)

    # Cell labels for smaller matrices only (avoids clutter)
    if matrix_size <= 15:
        for i in range(matrix_size):
            for j in range(matrix_size):
                value = correlation_matrix.iat[i, j]
                text_color = 'white' if abs(value) > 0.55 else 'black'
                ax.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=8, color=text_color)

    colorbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('Correlation')

    fig.tight_layout()
    safe_pyplot(fig)
    plt.close(fig)