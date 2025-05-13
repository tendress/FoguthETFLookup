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

    # Function to Calculate Returns
    def calculate_returns(df):
        returns = {}
        for symbol in df['symbol'].unique():
            if symbol == 'Volatility Index':  # Skip calculating returns for ^VIX
                continue
            symbol_data = df[df['symbol'] == symbol]
            if not symbol_data.empty:
                start_price = symbol_data.iloc[0]['Close']
                end_price = symbol_data.iloc[-1]['Close']
                returns[symbol] = ((end_price - start_price) / start_price) * 100
        return returns

    # Function to Plot Stock Market Indicators
    def plot_stock_market_indicators(db_path, start_date, end_date):
        conn = sqlite3.connect(db_path)
        query = """
        SELECT Date, symbol, Close
        FROM etf_historical_prices
        WHERE symbol IN ('^DJI', '^GSPC', '^IXIC', '^VIX')
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df['Date'] = pd.to_datetime(df['Date'])

        # Map symbols to their full names
        symbol_mapping = {
            '^DJI': 'Dow Jones Industrial Average',
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^VIX': 'Volatility Index'
        }
        df['symbol'] = df['symbol'].map(symbol_mapping)

        # Filter data based on user-selected date range
        df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

        # Calculate returns
        returns = calculate_returns(df)
        # Add a divider
        st.markdown("<hr>", unsafe_allow_html=True)
        # Header for "Stock Market Indicators"
        st.header("Stock Market Indicators")
        
        
        st.markdown(
        "<span style='color:#0066CC; font-weight:bold;'>The Dow Jones Industrial Average measures 30 major U.S. companies, providing insight into the health of established industries and overall economic stability.</span>",
        unsafe_allow_html=True)  
        st.markdown(
        "<span style='color:#6699FF; font-weight:bold;'>The S&P 500 tracks the performance of the 500 large U.S. companies across various sectors, reflecting broad market trends and investor confidence in the U.S. Economy</span>",
        unsafe_allow_html=True)    
        st.markdown(
        "<span style='color:#FF3333; font-weight:bold;'>The NASDAQ focuses on technology and growth-oriented companies, indicating investor sentiment toward innovation and high-growth sectors.</span>",
        unsafe_allow_html=True)    
         
        
        
        
        # Display the most recent value for each stock market index
        
        # Display returns
        st.subheader("Stock Market Returns")
        for symbol, return_value in returns.items():
            st.write(f"{symbol}: {return_value:.2f}%")

        # Display the most recent value for ^VIX
        vix_data = df[df['symbol'] == 'Volatility Index']
        if not vix_data.empty:
            most_recent_vix = vix_data.iloc[-1]['Close']
            st.write(f"Volatility Index (VIX): {most_recent_vix:.2f}")
            

        # Separate data for Volatility Index (^VIX)
        other_data = df[df['symbol'] != 'Volatility Index']



        # Plot SP500, DJIA, NASDAQ in one chart
        fig1 = px.line(other_data, x='Date', y='Close', color='symbol',
                    title='Stock Market Indicators: Dow Jones, S&P 500, NASDAQ')
        st.plotly_chart(fig1, use_container_width=True)

        # Plot Volatility Index in its own chart
        st.markdown(
            "<span style='color:#FF3333; font-weight:bold;'>The VIX, or CBOE Volatility Index, measures market expectations of near-term volatility in the S&P 500, indicating investor fear or uncertainty. Higher VIX values suggest greater market instability, while lower imply calmer conditions, derived from demand for out-of-the-money S&P 500 options, which investors buy as protection against market uncertainty and potential downturns. A normal VIX value ranges between 12 and 20. </span>",
            unsafe_allow_html=True)    
        fig2 = px.line(vix_data, x='Date', y='Close', color='symbol',
                    title='Volatility Index (VIX)')
        st.plotly_chart(fig2, use_container_width=True)

    # Function to Plot International Market Indicators
    def plot_international_market_indicators(db_path, start_date, end_date):
        conn = sqlite3.connect(db_path)
        query = """
        SELECT Date, symbol, Close
        FROM etf_historical_prices
        WHERE symbol IN ('^N225', '^FTSE', '^DJSH')
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df['Date'] = pd.to_datetime(df['Date'])

        # Map symbols to their full names
        symbol_mapping = {
            '^N225': 'Nikkei 225',
            '^FTSE': 'FTSE 100',
            '^DJSH': 'Dow Jones Shanghai'
        }
        df['symbol'] = df['symbol'].map(symbol_mapping)

        # Filter data based on user-selected date range
        df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

        # Calculate returns
        returns = calculate_returns(df)

        # Add a divider
        st.markdown("<hr>", unsafe_allow_html=True)
        # Header for "International Market Indicators"
        st.header("International Market Indicators")

        # Display returns
        st.subheader("International Market Returns")
        for symbol, return_value in returns.items():
            st.write(f"{symbol}: {return_value:.2f}%")

        
        st.markdown(
        "<span style='color:red; font-weight:bold;'>Dow Jones Shanghai Index currently Not Reporting Data.</span>",
        unsafe_allow_html=True)    # Plot International Market Indicators
        fig = px.line(df, x='Date', y='Close', color='symbol',
                    title='International Market Indicators: Nikkei 225, FTSE 100, Dow Jones Shanghai')
        st.plotly_chart(fig, use_container_width=True)


    def plot_bond_yields(db_path, start_date, end_date):
        conn = sqlite3.connect(db_path)
        query = """
        SELECT Date, symbol, Close
        FROM etf_historical_prices
        WHERE symbol IN ('^IRX', '^FVX', '^TNX', '^TYX')
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df['Date'] = pd.to_datetime(df['Date'])

        # Map symbols to their full names
        symbol_mapping = {
            '^IRX': '3-Month Treasury Yield',
            '^FVX': '5-Year Treasury Yield',
            '^TNX': '10-Year Treasury Yield',
            '^TYX': '30-Year Treasury Yield'
        }
        df['symbol'] = df['symbol'].map(symbol_mapping)

        # Filter data based on user-selected date range
        df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

        # Get the data for economic_indicators.symbol = 'T10Y2Y' and graph it separately
        conn = sqlite3.connect(db_path)
        query = """
        SELECT Date, symbol, economic_value AS Close
        FROM economic_indicators
        WHERE symbol = 'T10Y2Y'
        """
        df2 = pd.read_sql_query(query, conn)
        conn.close()
        df2['Date'] = pd.to_datetime(df2['Date'])
        df2 = df2[(df2['Date'] >= pd.Timestamp(start_date)) & (df2['Date'] <= pd.Timestamp(end_date))]
        df2['symbol'] = 'T10Y2Y'

        df2['symbol'] = df2['symbol'].replace({'T10Y2Y': '10-Year Minus 2-Year Treasury Yield'})

        # Adding a divider
        st.markdown("<hr>", unsafe_allow_html=True)


        # Header for "Bond Yields"
        st.header("Bond Yields")

        st.markdown(
        "<span style='color:#3399FF; font-weight:bold;'>The 3-Month Treasury Bill yield reflects short-term investor confidence and expectations for monetary policy. Higher yields often suggest tighter policy or economic optimism, while lower yields may indicate economic caution or expectations of rate cuts.</span>",
        unsafe_allow_html=True)  
        st.markdown(
        "<span style='color:#CC0000; font-weight:bold;'>The 5-Year Treasury Note yield reflects intermediate-term investor expectations for economic growth and inflation, with higher yields suggesting confidence in moderate economic expansion but also concerns about persistent inflation or tighter monetary policy</span>",
        unsafe_allow_html=True)
        st.markdown(
        "<span style='color:#FF6666; font-weight:bold;'>The 10-Year Treasury Note yield indicates long-term investor expectations for the U.S. economy, with elevated yields suggesting cautious optimism for growth but also concerns about persistent inflation, trade policy uncertainties, and rising federal debt levels.</span>",
        unsafe_allow_html=True)
        st.markdown(
        "<span style='color:#0066CC; font-weight:bold;'>The 30-Year Treasury Bond yield indicates long-term investor expectations for the U.S. economy, higher yields indicating anticipation of sustained economic growth or rising inflation, but also signaling concerns about long-term fiscal challenges, such as increasing federal debt and potential trade disruptions.</span>",
        unsafe_allow_html=True)

        # Plot Bond Yields Over Time (excluding 10-Year Minus 2-Year Treasury Yield)
        bond_yields_df = df[df['symbol'] != '10-Year Minus 2-Year Treasury Yield']
        fig = px.line(bond_yields_df, x='Date', y='Close', color='symbol',
                    title='Bond Yields: 3-Month, 5-Year, 10-Year, 30-Year')
        st.plotly_chart(fig, use_container_width=True)

        # Display a graph of the 10-Year Minus 2-Year Treasury Yield
        fig2 = px.line(df2, x='Date', y='Close',
                    title='10-Year Minus 2-Year Treasury Yield')
        fig2.update_traces(mode='lines+markers', marker=dict(size=1), line=dict(width=1))  # Reduce line thickness

        # Add a horizontal line at y=0
        fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Line", 
                    annotation_position="top left")

        fig2.update_layout(xaxis_title="Date", yaxis_title="Yield (%)")
        st.plotly_chart(fig2, use_container_width=True)

        # Display the most recent value for 10-Year Minus 2-Year Treasury Yield
        if not df2.empty:
            most_recent_t10y2y = df2.iloc[-1]['Close']
            st.write(f"10-Year Minus 2-Year Treasury Yield: {most_recent_t10y2y:.2f}%")
            st.markdown(
            "<span style='color:#0066CC; font-weight:bold;'>The difference between the 10-year and the 2-year Treasury yields, known as the Yield Curve Spread, indicates investor expectations about future economic growth and monetary policy. A positive spread suggests cautious optimism for economic expansion while a negative spread (inverted yield curve) often signals recession risk.</span>",
            unsafe_allow_html=True)

        # User selects a specific date
        st.subheader("Select a Date to View Bond Yields")
        unique_dates = bond_yields_df['Date'].dt.date.unique()
        selected_date = st.date_input("Select a Date", value=unique_dates[-1], min_value=min(unique_dates), max_value=max(unique_dates))

        # Filter data for the selected date
        selected_date_data = bond_yields_df[bond_yields_df['Date'].dt.date == selected_date]

        # Ensure the order of symbols is 3-Month, 5-Year, 10-Year, 30-Year
        order = ['3-Month Treasury Yield', '5-Year Treasury Yield', '10-Year Treasury Yield', '30-Year Treasury Yield']
        selected_date_data['symbol'] = pd.Categorical(selected_date_data['symbol'], categories=order, ordered=True)
        selected_date_data = selected_date_data.sort_values('symbol')

        # Create a line chart for the selected date
        fig_selected_date = px.line(selected_date_data, x='symbol', y='Close', markers=True,
                                    title=f'Bond Yields for {selected_date}')
        fig_selected_date.update_traces(mode='lines+markers', marker=dict(size=10))
        fig_selected_date.update_layout(xaxis_title="Bond Type", yaxis_title="Yield (%)")
        st.plotly_chart(fig_selected_date, use_container_width=True)

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

        for symbol, title in indicators.items():
            # Fetch data for the current indicator
            query = f"""
            SELECT Date, economic_value AS Close
            FROM economic_indicators
            WHERE symbol = '{symbol}'
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

        for symbol, title in indicators.items():
            # Fetch data for the current indicator
            query = f"""
            SELECT Date, economic_value AS Close
            FROM economic_indicators
            WHERE symbol = '{symbol}'
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

    def plot_us_consumer(db_path, start_date, end_date):
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # List of U.S. Consumer indicators to graph
        indicators = {
            'UMCSENT': 'University of Michigan Consumer Sentiment Index',
            'MRTSSM44X72USS': 'Retail Sales (Excluding Food Services)',
            'MORTGAGE30US': '30-Year Fixed Mortgage Rate',
            'PAYEMS': 'Total Nonfarm Payrolls'
        }

        # Header for "U.S. Consumer"
        st.header("U.S. Consumer")

        for symbol, title in indicators.items():
            # Fetch data for the current indicator
            query = f"""
            SELECT Date, economic_value AS Close
            FROM economic_indicators
            WHERE symbol = '{symbol}'
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

    def plot_custom_chart(db_path, start_date, end_date):
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # Fetch available symbols and their names from the database
        query = """
        SELECT symbol, name
        FROM economic_indicators
        UNION
        SELECT symbol, name
        FROM etfs
        """
        symbols_df = pd.read_sql_query(query, conn)

        # Create a dictionary mapping symbols to their names for display
        symbol_name_mapping = dict(zip(symbols_df['symbol'], symbols_df['name']))

        # Sidebar for selecting symbols to plot
        st.sidebar.header("Custom Chart")
        selected_names = st.sidebar.multiselect(
            "Select Economic Indicators or ETFs to Plot",
            options=symbols_df['name'].tolist(),
            default=[]
        )

        # Map selected names back to their symbols
        selected_symbols = [symbol for symbol, name in symbol_name_mapping.items() if name in selected_names]

        # Fetch data for the selected symbols
        if selected_symbols:
            data_frames = []
            for symbol in selected_symbols:
                query = f"""
                SELECT Date, economic_value AS Close, '{symbol}' AS symbol
                FROM economic_indicators
                WHERE symbol = '{symbol}'
                UNION
                SELECT Date, Close, '{symbol}' AS symbol
                FROM etf_historical_prices
                WHERE symbol = '{symbol}'
                """
                df = pd.read_sql_query(query, conn)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
                data_frames.append(df)

            # Combine all data into a single DataFrame
            if data_frames:
                combined_df = pd.concat(data_frames)

                # Create a subplot with secondary Y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for i, symbol in enumerate(selected_symbols):
                    symbol_data = combined_df[combined_df['symbol'] == symbol]
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_data['Date'],
                            y=symbol_data['Close'],
                            mode='lines+markers',
                            name=symbol_name_mapping[symbol]  # Use the name for the legend
                        ),
                        secondary_y=(i % 2 == 1)  # Alternate Y-axes
                    )

                # Update layout
                fig.update_layout(
                    title_text="Custom Chart: Selected Economic Indicators and ETFs",
                    xaxis_title="Date",
                    yaxis_title="Primary Y-Axis",
                    yaxis2_title="Secondary Y-Axis",
                    legend_title="Symbols",
                    showlegend=True
                )
                fig.update_traces(mode='lines+markers', marker=dict(size=1), line=dict(width=2))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data available for the selected symbols in the chosen timeframe.")
        else:
            st.write("Select symbols from the sidebar to plot a custom chart.")

        # Close the database connection
        conn.close()


    # Run the Functions if the Database Path is Provided
    try:
        # Plot Stock Market Indicators
        plot_stock_market_indicators(db_path, start_date, end_date)

        # Plot Bond Yields
        plot_bond_yields(db_path, start_date, end_date)

        # Plot International Market Indicators
        plot_international_market_indicators(db_path, start_date, end_date)
        
        # Plot Federal Reserve Indicators
        plot_federal_reserve_indicators(db_path, start_date, end_date)

        # Plot U.S. Economy Indicators
        plot_us_economy(db_path, start_date, end_date)
        
        # Plot U.S. Consumer Indicators
        plot_us_consumer(db_path, start_date, end_date)
        
        # Plot Custom Chart
        plot_custom_chart(db_path, start_date, end_date)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Call the main function
if __name__ == "__main__":
    economic_indicators()