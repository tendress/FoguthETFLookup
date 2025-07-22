import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, date

# Set Streamlit page configuration
st.set_page_config(page_title="Sector Rotation Research", layout="wide")

def get_xl_etfs():
    """
    Get all ETFs starting with 'XL' from the etfs table.
    Returns a DataFrame with id, symbol, and name.
    """
    conn = sqlite3.connect("foguth_etf_models.db")
    query = """
    SELECT id, symbol, name
    FROM etfs
    WHERE symbol LIKE 'XL%'
    ORDER BY symbol
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def calculate_time_weighted_return(etf_symbol, start_date=None, end_date=None):
    """
    Calculate time-weighted return for a specific ETF using price data.
    Returns a DataFrame with Date and cumulative return.
    """
    conn = sqlite3.connect("foguth_etf_models.db")
    
    # Build the query with optional date filtering
    query = """
    SELECT Date, Close
    FROM etf_prices
    WHERE symbol = ?
    ORDER BY Date ASC
    """
    
    df = pd.read_sql_query(query, conn, params=(etf_symbol,))
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by date range if provided
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()
    
    # Calculate cumulative return (time-weighted)
    df['cum_return'] = ((1 + df['daily_return']).cumprod() - 1) * 100
    
    # Fill NaN values in cumulative return (first row will be NaN)
    df['cum_return'] = df['cum_return'].fillna(0)
    
    return df[['Date', 'cum_return', 'Close']].copy()

def plot_time_weighted_returns(selected_etfs, start_date, end_date):
    """
    Plot time-weighted returns for selected ETFs.
    """
    if not selected_etfs:
        st.warning("Please select at least one ETF to plot.")
        return
    
    # Prepare data for plotting
    plot_data = []
    
    for etf in selected_etfs:
        returns_df = calculate_time_weighted_return(etf, start_date, end_date)
        if not returns_df.empty:
            returns_df['ETF'] = etf
            plot_data.append(returns_df[['Date', 'cum_return', 'ETF']])
        else:
            st.warning(f"No data available for {etf} in the selected date range.")
    
    if not plot_data:
        st.error("No data available for any selected ETFs.")
        return
    
    # Combine all data
    combined_df = pd.concat(plot_data, ignore_index=True)
    
    # Create the plot
    fig = px.line(
        combined_df,
        x='Date',
        y='cum_return',
        color='ETF',
        title=f'Time-Weighted Returns: {", ".join(selected_etfs)}',
        labels={'cum_return': 'Cumulative Return (%)', 'Date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display performance summary
    st.subheader("Performance Summary")
    summary_data = []
    
    for etf in selected_etfs:
        returns_df = calculate_time_weighted_return(etf, start_date, end_date)
        if not returns_df.empty:
            total_return = returns_df['cum_return'].iloc[-1]
            start_price = returns_df['Close'].iloc[0]
            end_price = returns_df['Close'].iloc[-1]
            
            summary_data.append({
                'ETF': etf,
                'Total Return (%)': total_return,  # Keep as number for sorting
                'Total Return Display': f"{total_return:.2f}%",  # String for display
                'Start Price': f"${start_price:.2f}",
                'End Price': f"${end_price:.2f}",
                'Price Change': f"${end_price - start_price:.2f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Sort by Total Return (%) in descending order (highest returns first)
        summary_df = summary_df.sort_values('Total Return (%)', ascending=False)
        
        # Drop the numeric column and rename the display column for final display
        display_df = summary_df.drop('Total Return (%)', axis=1)
        display_df = display_df.rename(columns={'Total Return Display': 'Total Return (%)'})
        
        st.dataframe(display_df, use_container_width=True)

def sector_rotation_research():
    # App title
    st.title("Sector Rotation Research Dashboard")
    st.markdown("**Analyze time-weighted returns for SPDR Sector ETFs (XL series)**")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Get XL ETFs
    try:
        xl_etfs_df = get_xl_etfs()
        
        if xl_etfs_df.empty:
            st.error("No XL ETFs found in the database.")
            return
        
        # Display available ETFs
        st.sidebar.subheader("Available XL ETFs")
        st.sidebar.dataframe(xl_etfs_df[['symbol', 'name']], use_container_width=True)
        
        # ETF selection
        st.sidebar.subheader("Select ETFs to Compare")
        etf_options = xl_etfs_df['symbol'].tolist()
        selected_etfs = st.sidebar.multiselect(
            "Choose ETFs:",
            options=etf_options,
            default=etf_options[:3] if len(etf_options) >= 3 else etf_options,
            help="Select one or more ETFs to compare their time-weighted returns"
        )
        
        # Date range selection
        st.sidebar.subheader("Date Range")
        
        # Get the earliest and latest dates available
        if selected_etfs:
            conn = sqlite3.connect("foguth_etf_models.db")
            date_query = """
            SELECT MIN(Date) as min_date, MAX(Date) as max_date
            FROM etf_prices
            WHERE symbol IN ({})
            """.format(','.join(['?' for _ in selected_etfs]))
            
            date_df = pd.read_sql_query(date_query, conn, params=selected_etfs)
            conn.close()
            
            if not date_df.empty and date_df['min_date'].iloc[0]:
                min_date = pd.to_datetime(date_df['min_date'].iloc[0]).date()
                max_date = pd.to_datetime(date_df['max_date'].iloc[0]).date()
            else:
                min_date = date(2000, 1, 1)
                max_date = date.today()
        else:
            min_date = date(2000, 1, 1)
            max_date = date.today()
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=date(max_date.year - 1, max_date.month, max_date.day),
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Quick date range buttons
        st.sidebar.subheader("Quick Date Ranges")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("YTD"):
                start_date = date(max_date.year, 1, 1)
                st.rerun()
        
        with col2:
            if st.button("1 Year"):
                start_date = date(max_date.year - 1, max_date.month, max_date.day)
                st.rerun()
        
        # Main content area
        if selected_etfs:
            # Display selected ETFs info
            selected_info = xl_etfs_df[xl_etfs_df['symbol'].isin(selected_etfs)]
            
            st.subheader("Selected ETFs")
            st.dataframe(selected_info, use_container_width=True)
            
            # Plot the returns
            st.subheader("Time-Weighted Returns Comparison")
            plot_time_weighted_returns(selected_etfs, start_date, end_date)
            
            # Individual ETF details
            if len(selected_etfs) == 1:
                st.subheader(f"Detailed Analysis: {selected_etfs[0]}")
                etf_data = calculate_time_weighted_return(selected_etfs[0], start_date, end_date)
                
                if not etf_data.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_return = etf_data['cum_return'].iloc[-1]
                        st.metric("Total Return", f"{total_return:.2f}%")
                    
                    with col2:
                        start_price = etf_data['Close'].iloc[0]
                        st.metric("Start Price", f"${start_price:.2f}")
                    
                    with col3:
                        end_price = etf_data['Close'].iloc[-1]
                        st.metric("End Price", f"${end_price:.2f}")
                    
                    # Show recent data
                    st.subheader("Recent Data")
                    st.dataframe(etf_data.tail(10), use_container_width=True)
        
        else:
            st.info("Please select at least one ETF from the sidebar to begin analysis.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure the database file 'foguth_etf_models.db' exists and contains the required tables.")

# Call the function to run the app
if __name__ == "__main__":
    sector_rotation_research()