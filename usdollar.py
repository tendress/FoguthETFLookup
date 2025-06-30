## Analyzing the strength of the US Dollar for the Financial Advisor ##
import sqlite3
import pandas as pd
from datetime import datetime, date
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Currencies", page_icon="🧮", layout="wide")

# Streamlit App Title
st.title("Global Currencies Analysis")

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

# Define all plotting functions here 
# All Currencies will be plotted with respect to the US Dollar value compared to the other currency
# Functions: USD/EUR, USD/CAD, USD/JPY, USD/GBP, USD/BTC


def plot_dollar_index(db_path, start_date, end_date):
    '''Plot the US Dollar Index (DXY) over time.'''
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT Date, Close, symbol
    FROM etf_prices
    WHERE symbol = 'DX-Y.NYB' AND Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Date;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(df, x='Date', y='Close', title='US Dollar Index (DXY)',
                  labels={'Close': 'Index Value', 'Date': 'Date'})
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Index Value')
    st.plotly_chart(fig, use_container_width=True)

def plot_usd_eur(db_path, start_date, end_date):
    '''Plot USD to EUR exchange rate over time.'''
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT Date, Close, symbol
    FROM etf_prices
    WHERE symbol = 'EURUSD=X' AND Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Date;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(df, x='Date', y='Close', title='US Dollar to Euro Exchange Rate',
                  labels={'Close': 'Exchange Rate', 'Date': 'Date'})
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Exchange Rate (USD/EUR)')
    st.plotly_chart(fig, use_container_width=True)
    
def plot_usd_cad(db_path, start_date, end_date):
    '''Plot USD to CAD exchange rate over time.'''
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT Date, Close, symbol
    FROM etf_prices
    WHERE symbol = 'CAD=X' AND Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Date;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(df, x='Date', y='Close', title='US Dollar to Canadian Dollar Exchange Rate',
                  labels={'Close': 'Exchange Rate', 'Date': 'Date'})
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Exchange Rate (USD/CAD)')
    st.plotly_chart(fig, use_container_width=True)

def plot_usd_jpy(db_path, start_date, end_date):
    '''Plot USD to JPY exchange rate over time.'''
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT Date, Close, symbol
    FROM etf_prices
    WHERE symbol = 'JPY=X' AND Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Date;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(df, x='Date', y='Close', title='US Dollar to Japanese Yen Exchange Rate',
                  labels={'Close': 'Exchange Rate', 'Date': 'Date'})
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Exchange Rate (USD/JPY)')
    st.plotly_chart(fig, use_container_width=True)
    
def plot_usd_gbp(db_path, start_date, end_date):
    '''Plot USD to GBP exchange rate over time.'''
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT Date, Close, symbol
    FROM etf_prices
    WHERE symbol = 'GBPUSD=X' AND Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Date;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(df, x='Date', y='Close', title='US Dollar to British Pounds Exchange Rate',
                  labels={'Close': 'Exchange Rate', 'Date': 'Date'})
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Exchange Rate (USD/GBP)')
    st.plotly_chart(fig, use_container_width=True)
    
def plot_usd_btc(db_path, start_date, end_date):
    '''Plot USD to BTC exchange rate over time.'''
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT Date, Close, symbol
    FROM etf_prices
    WHERE symbol = 'BTC-USD' AND Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Date;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(df, x='Date', y='Close', title='US Dollar to Bitcoin Exchange Rate',
                  labels={'Close': 'Exchange Rate', 'Date': 'Date'})
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Exchange Rate (USD/BTC)')
    st.plotly_chart(fig, use_container_width=True)
    
# Main function to run the Streamlit app
def main():
    plot_dollar_index(db_path, start_date, end_date)
    plot_usd_eur(db_path, start_date, end_date)
    plot_usd_cad(db_path, start_date, end_date)
    plot_usd_jpy(db_path, start_date, end_date)
    plot_usd_gbp(db_path, start_date, end_date)
    plot_usd_btc(db_path, start_date, end_date)
    
if __name__ == "__main__":
    main()