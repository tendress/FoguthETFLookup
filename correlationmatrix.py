import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
import datetime  # Import datetime for date calculations

def display_correlation_matrix():
    st.title("ETF Correlation Matrix")
    st.write("This page displays the correlation matrix for ETFs based on their historical prices.")

    # Create a SQLite database connection
    database_path = 'foguth_etf_models.db'  # Replace with your database path
    conn = sqlite3.connect(database_path)

    # Fetch the list of models
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM models")
    models = [row[0] for row in cursor.fetchall()]

    if not models:
        st.warning("No models found in the database.")
        conn.close()
        return

    # Let the user select a model
    selected_model = st.sidebar.selectbox("Select a Model", models)

    # Fetch the ETFs associated with the selected model
    query = """
        SELECT e.symbol
        FROM etfs e
        JOIN security_sets_etfs sse ON e.id = sse.etf_id
        JOIN security_sets ss ON sse.security_set_id = ss.id
        JOIN model_security_set mss ON ss.id = mss.security_set_id
        JOIN models m ON mss.model_id = m.id
        WHERE m.name = ?
    """
    cursor.execute(query, (selected_model,))
    etf_symbols = [row[0] for row in cursor.fetchall()]

    if not etf_symbols:
        st.warning(f"No ETFs found for the selected model: {selected_model}.")
        conn.close()
        return

    # Let the user select a date range (default to Year to Date)
    st.sidebar.header("Filter by Date Range")
    today = datetime.date.today()
    start_of_year = datetime.date(today.year, 1, 1)
    start_date = st.sidebar.date_input("Start Date", value=start_of_year, key="start_date")
    end_date = st.sidebar.date_input("End Date", value=today, key="end_date")

    if start_date > end_date:
        st.error("Start date must be before end date.")
        conn.close()
        return

    # Fetch price data for the ETFs from the etf_prices table
    query = """
        SELECT symbol, Date, Close
        FROM etf_prices
        WHERE symbol IN ({}) AND Date BETWEEN ? AND ?
        ORDER BY Date ASC
    """.format(','.join(['?'] * len(etf_symbols)))

    price_data = pd.read_sql_query(query, conn, params=etf_symbols + [start_date, end_date])

    # Close the database connection
    conn.close()

    if price_data.empty:
        st.warning("No price data found for the selected ETFs and date range.")
        return

    # Pivot the data to create a DataFrame with symbols as columns and dates as the index
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    pivoted_data = price_data.pivot(index='Date', columns='symbol', values='Close')

    # Calculate the correlation matrix
    correlation_matrix = pivoted_data.corr()

    # Create an interactive heatmap using Plotly
    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale='RdBu',
        title=f"ETF Correlation Matrix ({selected_model})",
        labels=dict(color="Correlation"),
    )

    # Update layout for better readability and add ETFs to the top x-axis
    fig.update_layout(
        xaxis_title="ETFs",
        yaxis_title="ETFs",
        width=2000,
        height=2000,
        xaxis=dict(tickangle=45),
        font=dict(size=24),
    )

    # Add a secondary x-axis (top axis) with the same ETF labels
    fig.update_layout(
        xaxis2=dict(
            tickmode='array',
            tickvals=list(range(len(correlation_matrix.columns))),
            ticktext=correlation_matrix.columns,
            side='top',
            tickangle=45,
        ),
    )

    # Link the top and bottom x-axes
    fig.update_layout(
        xaxis=dict(matches='x2'),
    )

    # Display the interactive Plotly graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)