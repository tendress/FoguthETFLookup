## Economic Indicators for the Financial Advisor ##
import sqlite3
import pandas as pd
from datetime import datetime, date
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit page configuration to wide layout
st.set_page_config(page_title="Economic Indicators Dashboard", layout="wide")

def economic_indicators():
    # Streamlit App Title
    st.title("Economic Indicators Dashboard")

    # Hardcoded database path
    db_path = "foguth_etf_models.db"

    # Sidebar for Date Range Selection
    st.sidebar.header("Select Time Frame")

    # Default start and end dates
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=datetime(1994, 1, 1), 
        min_value=datetime(1994, 1, 1), 
        max_value=datetime.today()
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        value=datetime.today(), 
        min_value=datetime(1994, 1, 1), 
        max_value=datetime.today()
    )

    # Add a button to set the timeframe to Year to Date (YTD)
    if st.sidebar.button("Set to Year to Date"):
        start_date = date(datetime.today().year, 1, 1)
        st.sidebar.success(f"Timeframe set to Year to Date: {start_date} to {end_date}")

    # Add buttons to set the timeframe
    if st.sidebar.button("Set to Last 3 Years"):
        start_date = date(datetime.today().year - 3, datetime.today().month, datetime.today().day)
        st.sidebar.success(f"Timeframe set to Last 3 Years: {start_date} to {end_date}")

    if st.sidebar.button("Set to Last 5 Years"):
        start_date = date(datetime.today().year - 5, datetime.today().month, datetime.today().day)
        st.sidebar.success(f"Timeframe set to Last 5 Years: {start_date} to {end_date}")

    if st.sidebar.button("Set to Last 10 Years"):
        start_date = date(datetime.today().year - 10, datetime.today().month, datetime.today().day)
        st.sidebar.success(f"Timeframe set to Last 10 Years: {start_date} to {end_date}")

    # Define all plotting functions here (already provided in your code)
    # Functions: plot_stock_market_indicators, plot_international_market_indicators,
    # plot_bond_yields, plot_federal_reserve_indicators, plot_us_economy,
    # plot_us_consumer, plot_custom_chart


    def plot_federal_reserve_indicators(db_path, start_date, end_date):
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # List of Federal Reserve indicators to graph
        indicators = {
            'FEDFUNDS': 'Federal Funds Rate',
            'UNRATE': 'Unemployment Rate',
            'T10YIE': '10-Year Breakeven Inflation Rate',  # Added T10YIE
            'CORESTICKM159SFRBATL': 'Core Sticky CPI'
        }

        # Header for "Federal Reserve"
        st.header("Federal Reserve")
        st.markdown(
            "<span style='color:#006699; font-weight:bold;'>The Federal Funds rate is the interest rate at which banks lend to each other overnight, set by the Federal Reserve, and it's critical because it influences borrowing costs, economic activity, and inflation across the U.S. economy.</span>",
            unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#006699; font-weight:bold;'>The Unemployment rate measures the percentage of the labor force that is jobless and actively seeking work.</span>",
            unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#006699; font-weight:bold;'>The 10-Year Breakeven Inflation rate, derived from the difference between the 10-year Treasury Note yield and the 10-Year Treasury Inflation-Protected Securities (TIPS) yield, reflects market expectations for average annual inflation over the next decade.</span>",
            unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#006699; font-weight:bold;'>Core Sticky CPI measures the inflation rate of less volatile consumer price components (excluding food and energy and focusing on prices that change slowly), provides a stable, long-term view of underlying inflation trends, helping policy-makers and investors gauge persistent inflationary pressures in the U.S. economy. </span>",
            unsafe_allow_html=True)
        
        
        for symbol, title in indicators.items():
            # Fetch data for the current indicator
            query = f"""
            SELECT Date, economic_value AS Close
            FROM economic_indicators
            WHERE symbol = '{symbol}'
            ORDER BY Date
            """
            df = pd.read_sql_query(query, conn)

            # Convert Date column to datetime and filter by date range
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

            # Plot the data if the DataFrame is not empty
            if not df.empty:
                fig = px.line(df, x='Date', y='Close', title=title)
                fig.update_traces(mode='lines+markers', marker=dict(size=1), line=dict(width=2))
                fig.update_layout(xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"No data available for {title} in the selected date range.")

        # Close the database connection
        conn.close()

    def plot_us_economy(db_path, start_date, end_date):
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # List of U.S. Economy indicators to graph
        indicators = {
            'GDP': 'Gross Domestic Product (GDP)',
            'WM2NS': 'M2 Money Stock',
            'PPIACO': 'Producer Price Index (PPI)'
        }

        # Header for "U.S. Economy"
        st.header("U.S. Economy")

        st.markdown(
            "<span style='color:#006699; font-weight:bold;'>GDP, Gross Domestic Product, is the total monetary value of all goods and services produced within a country over a specific period, and it measures a nation's economic performance and health.</span>",
            unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#006699; font-weight:bold;'>M2 is a measure of the money supply that includes cash, checking deposits, and easily convertible near money like savings accounts and money market funds and helps gauge future inflation, economic growth and the effectiveness of monetary policy. </span>",
            unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#006699; font-weight:bold;'>The Producer Price Index, PPI, measures the average change over time in the selling prices received by domestic producers for their goods and services, and indicates inflation trends at the wholesale level and can signal future consumer price changes. </span>",
            unsafe_allow_html=True)
        
        for symbol, title in indicators.items():
            # Fetch data for the current indicator
            query = f"""
            SELECT Date, economic_value AS Close
            FROM economic_indicators
            WHERE symbol = '{symbol}'
            ORDER BY Date
            """
            df = pd.read_sql_query(query, conn)

            # Convert Date column to datetime and filter by date range
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

            # Plot the data if the DataFrame is not empty
            if not df.empty:
                fig = px.line(df, x='Date', y='Close', title=title)
                fig.update_traces(mode='lines+markers', marker=dict(size=1), line=dict(width=2))
                fig.update_layout(xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"No data available for {title} in the selected date range.")

        # Close the database connection
        conn.close()



    # Run the Functions if the Database Path is Provided
    try:

        
        # Plot Federal Reserve Indicators
        plot_federal_reserve_indicators(db_path, start_date, end_date)

        # Plot U.S. Economy Indicators
        plot_us_economy(db_path, start_date, end_date)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Call the main function
if __name__ == "__main__":
    economic_indicators()