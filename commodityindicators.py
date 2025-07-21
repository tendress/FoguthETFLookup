## Consumer Indicators for the Financial Advisor ##
import sqlite3
import pandas as pd
from datetime import datetime, date
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit page configuration to wide layout
st.set_page_config(page_title="Economic Indicators Dashboard", layout="wide")

def commodity_indicators():
    # Streamlit App Title
    st.title("Economic Indicators Dashboard")

    # Updated database path to use FRED indicators database
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
 
    def plot_commodities(db_path, start_date, end_date):
        # Connect to the FRED indicators database
        conn = sqlite3.connect(db_path)

        # List of commodity indicators from FRED to graph
        commodities = {
            'PPIACO': 'Producer Price Index by Commodity: All Commodities',
            'WTISPLC': 'Spot Crude Oil Price: West Texas Intermediate (WTI)',
            'GOLDAMGBD228NLBM': 'Gold Fixing Price',
            'PCOPPUSDM': 'Global Price of Copper'
        }

        # Header for "Commodities"
        st.header("Commodity Indicators")

        for symbol, title in commodities.items():
            # Updated query to use the FRED database structure
            query = f"""
            SELECT IV.date AS Date, IV.value AS Close
            FROM indicator_values IV
            INNER JOIN economic_indicators EI ON IV.economic_indicator_id = EI.id
            WHERE EI.symbol = '{symbol}'
            ORDER BY IV.date
            """
            df = pd.read_sql_query(query, conn)

            # Convert Date column to datetime and filter by date range
            if not df.empty:
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
            else:
                st.write(f"No data found for {title}.")

        # Close the database connection
        conn.close()
        
    def plot_fallback_commodities(start_date, end_date):
        # Fallback to ETF database for futures data if FRED commodity data is not available
        etf_db_path = "foguth_etf_models.db"
        
        try:
            conn = sqlite3.connect(etf_db_path)
            
            # List of commodity futures from ETF database
            fallback_commodities = {
                'GC=F': 'Gold Futures',
                'CL=F': 'Crude Oil Futures',
                'HG=F': 'Copper Futures'
            }

            st.subheader("Commodity Futures")

            for symbol, title in fallback_commodities.items():
                # Fetch data for the current commodity
                query = f"""
                SELECT Date, Close
                FROM etf_prices
                WHERE symbol = '{symbol}'
                ORDER BY Date
                """
                df = pd.read_sql_query(query, conn)

                # Convert Date column to datetime and filter by date range
                if not df.empty:
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
                else:
                    st.write(f"No data found for {title}.")

            conn.close()
            
        except Exception as e:
            st.error(f"Error accessing fallback commodity data: {e}")
        
    try:  
        # Plot Commodities from FRED database
        plot_commodities(db_path, start_date, end_date)
        
        # Also plot fallback commodity futures data
        plot_fallback_commodities(start_date, end_date)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
# Call the main function
if __name__ == "__main__":
    commodity_indicators()