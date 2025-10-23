import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def benchmark_returns_dashboard():
    # Page config
    st.set_page_config(page_title="Benchmark Returns Dashboard", layout="wide")
    st.title("Daily Benchmark Returns Dashboard")
    
    # Connect to database
    @st.cache_resource
    def get_connection():
        return sqlite3.connect("foguthbenchmarks.db", check_same_thread=False)
    
    conn = get_connection()
    
    # Sidebar for filtering
    st.sidebar.header("Filter Options")
    
    # Get list of strategies
    @st.cache_data(ttl=3600)
    def get_strategies():
        query = "SELECT DISTINCT strategy_benchmark FROM benchmark_returns"
        return pd.read_sql(query, conn)["strategy_benchmark"].tolist()
    
    strategies = get_strategies()
    
    # Strategy selector
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies", 
        options=strategies,
        default=strategies[:3] if len(strategies) >= 3 else strategies
    )
    
    # Date range selection
    date_options = {
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 180 Days": 180,
        "Last 1 Year": 365,
        "Last 3 Years": 365 * 3,
        "Last 5 Years": 365 * 5,
        "All Time": None
    }
    
    date_range = st.sidebar.selectbox("Select Date Range", options=list(date_options.keys()))
    
    # Calculate start date based on selection
    days = date_options[date_range]
    if days:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    else:
        start_date = "1900-01-01"  # Effectively all data
        
    # Advanced options
    st.sidebar.header("Display Options")
    show_rolling_avg = st.sidebar.checkbox("Show Rolling Average", value=True)
    rolling_window = st.sidebar.slider("Rolling Window (days)", 5, 90, 21) if show_rolling_avg else 0
    
    # Calculate cumulative returns
    show_cumulative = st.sidebar.checkbox("Show Cumulative Returns", value=False)
    
    # Main content area - display returns
    if not selected_strategies:
        st.warning("Please select at least one strategy to display.")
    else:
        # Fetch data
        placeholders = ','.join(['?' for _ in selected_strategies])
        query = f"""
        SELECT date, strategy_benchmark, return 
        FROM benchmark_returns
        WHERE strategy_benchmark IN ({placeholders})
        AND date >= ?
        ORDER BY date
        """
        
        params = selected_strategies + [start_date]
        df = pd.read_sql(query, conn, params=params)
        
        if df.empty:
            st.warning("No data found for the selected criteria.")
        else:
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Line Chart", "Heatmap"])
            
            with tab1:
                # Create either daily returns or cumulative returns
                if show_cumulative:
                    # Create a pivot table for easier calculation
                    pivot_df = df.pivot(index='date', columns='strategy_benchmark', values='return')
                    
                    # Calculate cumulative returns
                    cum_returns = (1 + pivot_df).cumprod() - 1
                    
                    # Melt back for plotting
                    cum_df = cum_returns.reset_index().melt(
                        id_vars='date',
                        var_name='strategy_benchmark',
                        value_name='cumulative_return'
                    )
                    
                    fig = px.line(
                        cum_df, 
                        x='date', 
                        y='cumulative_return',
                        color='strategy_benchmark',
                        title='Cumulative Benchmark Returns',
                        labels={'cumulative_return': 'Cumulative Return', 'date': 'Date'}
                    )
                    
                    # Format y-axis as percentage
                    fig.update_layout(yaxis_tickformat='.1%')
                    
                else:
                    # Create line chart of daily returns
                    fig = px.line(
                        df, 
                        x='date', 
                        y='return',
                        color='strategy_benchmark',
                        title='Daily Benchmark Returns',
                        labels={'return': 'Daily Return', 'date': 'Date'}
                    )
                    
                    # Format y-axis as percentage
                    fig.update_layout(yaxis_tickformat='.1%')
                    
                    # Add rolling average if requested
                    if show_rolling_avg:
                        for strategy in selected_strategies:
                            strategy_data = df[df['strategy_benchmark'] == strategy]
                            rolling_avg = strategy_data.set_index('date')['return'].rolling(rolling_window).mean()
                            
                            fig.add_trace(go.Scatter(
                                x=rolling_avg.index,
                                y=rolling_avg,
                                mode='lines',
                                line=dict(width=3, dash='dash'),
                                name=f"{strategy} ({rolling_window}-day MA)"
                            ))
                
                # Enhance the graph appearance
                fig.update_layout(
                    height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40),
                    hovermode="x unified",
                    xaxis_title="Date",
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics section
                st.subheader("Performance Statistics")
                
                # Create pivot table for statistics calculation
                pivot = df.pivot(index='date', columns='strategy_benchmark', values='return')
                
                # Calculate statistics
                stats = pd.DataFrame({
                    'Mean Return': pivot.mean(),
                    'Annualized Return': pivot.mean() * 252,  # Approximate trading days in a year
                    'Std Dev': pivot.std(),
                    'Annualized Volatility': pivot.std() * np.sqrt(252),
                    'Sharpe Ratio': (pivot.mean() * 252) / (pivot.std() * np.sqrt(252)),
                    # Fix: Calculate max drawdown correctly (use .max() to get a single value per strategy)
                    'Max Drawdown': ((pivot.cummax() - pivot) / pivot.cummax()).max(),
                    'Best Day': pivot.max(),
                    'Worst Day': pivot.min()
                })
                
                # Format the stats
                stats_display = stats.copy()
                for col in ['Mean Return', 'Annualized Return', 'Best Day', 'Worst Day']:
                    stats_display[col] = stats_display[col].map('{:.2%}'.format)
                
                for col in ['Std Dev', 'Annualized Volatility']:
                    stats_display[col] = stats_display[col].map('{:.2%}'.format)
                
                for col in ['Sharpe Ratio']:
                    stats_display[col] = stats_display[col].map('{:.2f}'.format)
                
                for col in ['Max Drawdown']:
                    stats_display[col] = stats_display[col].map('{:.2%}'.format)
                
                st.dataframe(stats_display, use_container_width=True)
            
            with tab2:
                # Create heatmap of returns
                pivot = df.pivot(index='date', columns='strategy_benchmark', values='return')
                pivot = pivot.resample('W').mean()  # Weekly resampling for better display
                
                fig = px.imshow(
                    pivot,
                    title="Weekly Average Returns Heatmap",
                    labels=dict(x="Strategy", y="Date", color="Return"),
                    color_continuous_scale="RdBu_r",
                    zmin=-0.02, zmax=0.02  # Setting reasonable bounds for color scale
                )
                
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    benchmark_returns_dashboard()