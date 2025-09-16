## Consumer Indicators for the Financial Advisor ##
import sqlite3
import pandas as pd
from datetime import datetime, date
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st



def consumer_indicators():
    # Streamlit App Title
    st.title("Economic Indicators Dashboard")

    # Hardcoded database path
    db_path = "foguth_fred_indicators.db"

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
 
    
    # Function to plot U.S. Consumer Indicators
    
    def plot_us_consumer(db_path, start_date, end_date):
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # List of U.S. Consumer indicators to graph
        indicators = {
            'UMCSENT': 'University of Michigan Consumer Sentiment Index',
            'CORESTICKM159SFRBATL': 'Core Sticky CPI',
            'MORTGAGE30US': '30-Year Fixed Mortgage Rate',
            'GASREGCOVW': 'Average Price: Regular Gasoline, U.S. (Weekly)',
            'MRTSSM44X72USS': 'Retail Sales (Excluding Food Services)',
            'MSPNHSUS': 'Median Sales Price of New Houses Sold in the U.S.',
            'HSN1F': 'New One Family Houses Sold: United States',
            'HNFSEPUSSA': 'New Privately-Owned Housing Units Started: Single-Family Units'
        }

        # Header for "U.S. Consumer"
        st.header("U.S. Consumer")

        for symbol, title in indicators.items():
            # Fetch data for the current indicator
            query = f"""
            SELECT date, economic_value AS Close
            FROM economic_indicators
            WHERE symbol = '{symbol}'
            ORDER BY date
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
        
    def plot_commodities(db_path, start_date, end_date):
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # List of commodities to graph
        commodities = {
            'GC=F': 'Gold Futures',
            'CL=F': 'Crude Oil Futures',
            'HG=F': 'Copper Futures'
        }

        # Header for "Commodities"
        st.header("Commodities")

        for symbol, title in commodities.items():
            # Fetch data for the current commodity
            query = f"""
            SELECT Date, Close
            FROM etf_prices
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
        
    try:  
        # Plot U.S. Consumer Indicators
        plot_us_consumer(db_path, start_date, end_date)
        
        # Plot Commodities
        #plot_commodities(db_path, start_date, end_date)  
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
# Call the main function
if __name__ == "__main__":
    consumer_indicators()
        
        