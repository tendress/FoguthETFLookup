## Analyzing the strength of the US Dollar for the Financial Advisor ##
import sqlite3
import pandas as pd
from datetime import datetime, date
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

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

# Define all plotting functions here 
# All Currencies will be plotted with respect to the US Dollar value compared to the other currency
# Functions: USD/EUR, USD/CAD, USD/JPY, USD/GBP, USD/BTC

def plot_usd_eur(db_path, start_date, end_date):