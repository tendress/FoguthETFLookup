
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
        FROM etf_prices
        WHERE symbol IN ('^DJI', '^GSPC', '^IXIC', '^VIX')
        ORDER BY Date
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

        # Display returns
        st.subheader("Stock Market Returns")
        for symbol, return_value in returns.items():
            st.write(f"{symbol}: {return_value:.2f}%")

        # Display the most recent value for ^VIX in a separate column on the right
        vix_data = df[df['symbol'] == 'Volatility Index']
        if not vix_data.empty:
            most_recent_vix = vix_data.iloc[-1]['Close']
            vix_color = "#FF3333" if most_recent_vix > 20 else "#000000"
            col1, col2, col3 = st.columns([2, 1, 2])
            with col3:
                st.markdown(
                    f"""
                    <div style="background-color:#FFFFF;padding:20px;border-radius:10px;text-align:center;">
                        <span style="font-size:22px;font-weight:bold;color:#000000;"><u>VIX (Volatility Index)</u></span><br>
                        <span style="font-size:32px;font-weight:bold;color:{vix_color};">{most_recent_vix:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )    
        # Normalize returns for each index (excluding VIX)
        other_data = df[df['symbol'] != 'Volatility Index'].copy()
        normalized_data = []
        for symbol in other_data['symbol'].unique():
            symbol_df = other_data[other_data['symbol'] == symbol].copy()
            if not symbol_df.empty:
                first_close = symbol_df.iloc[0]['Close']
                symbol_df['Normalized'] = (symbol_df['Close'] / first_close) * 100  # Start at 100
                normalized_data.append(symbol_df)
        if normalized_data:
            normalized_df = pd.concat(normalized_data)
            fig1 = px.line(
                normalized_df,
                x='Date',
                y='Normalized',
                color='symbol',
                title='Normalized Stock Market Indicators: Dow Jones, S&P 500, NASDAQ (Start = 100)'
            )
            fig1.update_layout(yaxis_title="Normalized Value (Start = 100)")
            st.plotly_chart(fig1, use_container_width=True)

        # Plot Volatility Index in its own chart
        st.markdown(
            "<span style='color:blue; font-weight:bold;'>The VIX, or CBOE Volatility Index, measures market expectations of near-term volatility in the S&P 500, indicating investor fear or uncertainty. Higher VIX values suggest greater market instability, while lower imply calmer conditions, derived from demand for out-of-the-money S&P 500 options, which investors buy as protection against market uncertainty and potential downturns. A normal VIX value ranges between 12 and 20. </span>",
            unsafe_allow_html=True)    
        fig2 = px.line(vix_data, x='Date', y='Close', color='symbol',
                    title='Volatility Index (VIX)')
        st.plotly_chart(fig2, use_container_width=True)

    
    
    def plot_bond_yields(db_path, start_date, end_date):
        conn = sqlite3.connect("foguth_fred_indicators.db")
        query = """
        SELECT IV.date, EI.symbol, IV.value
        FROM indicator_values IV
        INNER JOIN economic_indicators EI ON IV.economic_indicator_id = EI.id
        WHERE EI.symbol IN ('DGS3MO', 'DGS2', 'DGS5', 'DGS10', 'DGS30')
        ORDER BY IV.date
        """
        df = pd.read_sql_query(query, conn)
        
        df['date'] = pd.to_datetime(df['date'])

        # Map symbols to their full names
        symbol_mapping = {
            'DGS3MO': '3-Month Treasury Yield',
            'DGS2': '2-Year Treasury Yield',
            'DGS5': '5-Year Treasury Yield',
            'DGS10': '10-Year Treasury Yield',
            'DGS30': '30-Year Treasury Yield'
        }
        df['symbol'] = df['symbol'].map(symbol_mapping)

        # Filter data based on user-selected date range
        df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
        
        # Rename columns for consistency
        df = df.rename(columns={'date': 'Date', 'value': 'Close'})

        # Get the data for T10Y2Y from the same database
        query2 = """
        SELECT IV.date AS Date, IV.value AS Close, EI.symbol
        FROM indicator_values IV
        INNER JOIN economic_indicators EI ON IV.economic_indicator_id = EI.id
        WHERE EI.symbol = 'T10Y2Y'
        ORDER BY IV.date
        """
        df2 = pd.read_sql_query(query2, conn)
        conn.close()
        
        if not df2.empty:
            df2['Date'] = pd.to_datetime(df2['Date'])
            df2 = df2[(df2['Date'] >= pd.Timestamp(start_date)) & (df2['Date'] <= pd.Timestamp(end_date))]
            df2['symbol'] = '10-Year Minus 2-Year Treasury Yield'

        # Adding a divider
        st.markdown("<hr>", unsafe_allow_html=True)

        # Header for "Bond Yields"
        st.header("Bond Yields")

        st.markdown(
        "<span style='color:#0066CC; font-weight:bold;'>The 3-Month Treasury Bill yield reflects short-term investor confidence and expectations for monetary policy. Higher yields often suggest tighter policy or economic optimism, while lower yields may indicate economic caution or expectations of rate cuts.</span>",
        unsafe_allow_html=True)
        st.markdown(
        "<span style='color:#3399FF; font-weight:bold;'>The 2-Year Treasury Note yield is closely watched as it reflects near-term Federal Reserve policy expectations and investor sentiment about economic conditions over the next two years, often moving in tandem with federal funds rate expectations.</span>",
        unsafe_allow_html=True)  
        st.markdown(
        "<span style='color:#CC0000; font-weight:bold;'>The 5-Year Treasury Note yield reflects intermediate-term investor expectations for economic growth and inflation, with higher yields suggesting confidence in moderate economic expansion but also concerns about persistent inflation or tighter monetary policy</span>",
        unsafe_allow_html=True)
        st.markdown(
        "<span style='color:#FF6666; font-weight:bold;'>The 10-Year Treasury Note yield indicates long-term investor expectations for the U.S. economy, with elevated yields suggesting cautious optimism for growth but also concerns about persistent inflation, trade policy uncertainties, and rising federal debt levels.</span>",
        unsafe_allow_html=True)
        st.markdown(
        "<span style='color:#009900; font-weight:bold;'>The 30-Year Treasury Bond yield indicates long-term investor expectations for the U.S. economy, higher yields indicating anticipation of sustained economic growth or rising inflation, but also signaling concerns about long-term fiscal challenges, such as increasing federal debt and potential trade disruptions.</span>",
        unsafe_allow_html=True)

        # Ensure the order of symbols includes 2-Year Treasury Yield
        order = ['3-Month Treasury Yield', '2-Year Treasury Yield', '5-Year Treasury Yield', '10-Year Treasury Yield', '30-Year Treasury Yield']
        bond_yields_df = df[df['symbol'].isin(order)].copy()
        bond_yields_df['symbol'] = pd.Categorical(bond_yields_df['symbol'], categories=order, ordered=True)
        bond_yields_df = bond_yields_df.sort_values(['Date', 'symbol'])

        # Plot Bond Yields Over Time (excluding 10-Year Minus 2-Year Treasury Yield)
        if not bond_yields_df.empty:
            fig = px.line(
                bond_yields_df,
                x='Date',
                y='Close',
                color='symbol',
                category_orders={'symbol': order},
                title='Bond Yields: 3-Month, 2-Year, 5-Year, 10-Year, 30-Year'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No bond yield data available for the selected date range.")

        # Display the most recent value for 10-Year Minus 2-Year Treasury Yield
        if not df2.empty:
            most_recent_t10y2y = df2.iloc[-1]['Close']
            st.subheader(f"10-Year Minus 2-Year Treasury Yield: {most_recent_t10y2y:.2f}%")
            st.markdown(
            "<span style='color:#0066CC; font-weight:bold;'>The difference between the 10-year and the 2-year Treasury yields, known as the Yield Curve Spread, indicates investor expectations about future economic growth and monetary policy. A positive spread suggests cautious optimism for economic expansion while a negative spread (inverted yield curve) often signals recession risk.</span>",
            unsafe_allow_html=True)

            # Display a graph of the 10-Year Minus 2-Year Treasury Yield
            fig2 = px.line(df2, x='Date', y='Close',
                        title='10-Year Minus 2-Year Treasury Yield')
            fig2.update_traces(mode='lines+markers', marker=dict(size=1), line=dict(width=1))  # Reduce line thickness

            # Add a horizontal line at y=0
            fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Line", 
                        annotation_position="top left")

            fig2.update_layout(xaxis_title="Date", yaxis_title="Yield (%)")
            st.plotly_chart(fig2, use_container_width=True)

        # User selects a specific date
        if not bond_yields_df.empty:
            st.subheader("The Yield Curve")
            st.markdown(
                "<span style='color:blue; font-weight:bold;'>A normal yield curve is upward-sloping, where longer-term Treasury yields are higher than shorter-term yields, reflecting investor expectations of economic growth and moderate inflation, with higher returns demanded for locking in funds over longer periods.</span>",
                unsafe_allow_html=True)
            unique_dates = bond_yields_df['Date'].dt.date.unique()
            selected_date = st.date_input("Select a Date", value=unique_dates[-1], min_value=min(unique_dates), max_value=max(unique_dates))

            # Filter data for the selected date
            selected_date_data = bond_yields_df[bond_yields_df['Date'].dt.date == selected_date].copy()

            # Ensure the order of symbols includes 2-Year Treasury Yield for the selected date
            selected_date_data.loc[:, 'symbol'] = pd.Categorical(selected_date_data['symbol'], categories=order, ordered=True)
            selected_date_data = selected_date_data.sort_values('symbol')

            # Create a line chart for the selected date
            if not selected_date_data.empty:
                fig_selected_date = px.line(
                    selected_date_data,
                    x='symbol',
                    y='Close',
                    markers=True,
                    category_orders={'symbol': order},
                    title=f'Bond Yields for {selected_date}'
                )
                fig_selected_date.update_traces(mode='lines+markers', marker=dict(size=10))
                fig_selected_date.update_layout(xaxis_title="Bond Type", yaxis_title="Yield (%)")
                st.plotly_chart(fig_selected_date, use_container_width=True)
            else:
                st.warning(f"No data available for {selected_date}")
    

    def plot_custom_chart(db_path, start_date, end_date):
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # Fetch available symbols and their names from the database
        query = """
        SELECT symbol, name
        FROM etfs
        WHERE name IS NOT NULL
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
                FROM etf_prices
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
         
        
        
                
        # Plot Custom Chart
        plot_custom_chart(db_path, start_date, end_date)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Call the main function
if __name__ == "__main__":
    economic_indicators()