### Foguth Financial ETF Lookup Tool ###
import pandas as pd
import streamlit as st
import sqlite3
import datetime
import plotly.express as px


def etf_lookup():
    st.title("ETF Lookup")
    st.write("This page allows you to look up ETF details.")

    # Database connection
    database_path = 'foguth_etf_models.db'
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch all ETFs
    @st.cache_data
    def load_all_etfs():
        cursor.execute('SELECT symbol FROM etfs')
        return [row[0] for row in cursor.fetchall()]

    # Fetch models and security sets
    @st.cache_data
    def load_models_and_security_sets():
        # Fetch all models
        cursor.execute('''
            SELECT DISTINCT models.name
            FROM model_security_set
            JOIN models ON model_security_set.model_id = models.id
        ''')
        models = [row[0] for row in cursor.fetchall()]

        # Fetch all security sets
        cursor.execute('''
            SELECT DISTINCT security_sets.name
            FROM security_sets
        ''')
        security_sets = [row[0] for row in cursor.fetchall()]

        return models, security_sets

    # Fetch security sets and ETFs for a selected model
    @st.cache_data
    def load_security_sets_and_etfs_for_model(selected_model):
        query = '''
            SELECT 
                security_sets.name AS security_set, 
                model_security_set.weight AS security_set_weight,
                etfs.symbol AS etf, 
                security_sets_etfs.weight AS etf_weight
            FROM model_security_set
            JOIN security_sets ON model_security_set.security_set_id = security_sets.id
            JOIN security_sets_etfs ON security_sets.id = security_sets_etfs.security_set_id
            JOIN etfs ON security_sets_etfs.etf_id = etfs.id
            JOIN models ON model_security_set.model_id = models.id
            WHERE models.name = ?
            AND security_sets_etfs.endDate IS NULL
            ORDER BY etf_weight DESC
        '''
        cursor.execute(query, (selected_model,))
        results = cursor.fetchall()
        return pd.DataFrame(results, columns=["Security Set", "Security Set Weight", "ETF", "ETF Weight"])

    # Load all ETFs
    all_etfs = load_all_etfs()

    # Load models and security sets
    models, security_sets = load_models_and_security_sets()

    # Sidebar: ETF selection (always show full list of ETFs)
    st.sidebar.title("Select an ETF")
    selected_etf = st.sidebar.selectbox(
        "Select an ETF",
        all_etfs,
        key="etf_selectbox"
    )

    # Sidebar: Filters
    st.sidebar.title("Filters")

    # Model filter
    selected_model = st.sidebar.selectbox(
        "Filter by Model",
        ["All Models"] + models,
        key="model_filter"
    )



    # Sidebar: Display security sets and ETFs for the selected model
    if selected_model != "All Models":
        st.sidebar.title(f"Model: {selected_model}")
        security_sets_and_etfs = load_security_sets_and_etfs_for_model(selected_model)

        if not security_sets_and_etfs.empty:
            for security_set in security_sets_and_etfs["Security Set"].unique():
                # Display security set name and weight
                security_set_weight = security_sets_and_etfs[
                    security_sets_and_etfs["Security Set"] == security_set
                ]["Security Set Weight"].iloc[0]
                st.sidebar.subheader(f"{security_set} ({security_set_weight*100}%)")

                # Display ETFs and their weights
                etfs_in_set = security_sets_and_etfs[
                    security_sets_and_etfs["Security Set"] == security_set
                ][["ETF", "ETF Weight"]]
                for _, row in etfs_in_set.iterrows():
                    st.sidebar.write(f"- {row['ETF']} ({row['ETF Weight']*100}%)")
        else:
            st.sidebar.write("No security sets or ETFs found for the selected model.")
    else:
        st.sidebar.write("Select a model to view its associated security sets and ETFs.")

# Main content: Display selected ETF information
    st.header(f"Details for Selected ETF: {selected_etf}")
    cursor.execute('SELECT * FROM etf_infos WHERE symbol = ?', (selected_etf,))
    result = cursor.fetchone()
    if result:
        # Get column names dynamically
        columns = [description[0] for description in cursor.description]
        etf_info = dict(zip(columns, result))

        st.markdown(f"### **{selected_etf} - {etf_info.get('longName', 'No name available')}**")
        st.write(f"**Category:** {etf_info.get('category', 'No category available')}")
        st.write(f"**Fund Manager:** {etf_info.get('fundFamily', 'No fund family available')}")
        st.write(f"**Dividend Yield:** {etf_info.get('dividendYield', 'No dividend yield available')}%")
        st.write(f"**Net Expense Ratio:** {etf_info.get('netExpenseRatio', 'No expense ratio available')}%")
        st.write(f"**Summary:** {etf_info.get('longBusinessSummary', 'No summary available.')}")
    else:
        st.write("No details available for the selected ETF.")

    # Display Top Holdings
    st.header("Top 10 Holdings")
    if result and etf_info.get('topHoldings'):
        try:
            # Parse the JSON string from the topHoldings column
            top_holdings = pd.read_json(etf_info['topHoldings'])
            top_holdings.index = range(1, len(top_holdings) + 1)
            top_holdings['Holding Percent'] = top_holdings['Holding Percent'].apply(lambda x: f"{x:.2f}")
            st.write(top_holdings)
        except Exception as e:
            st.write("Unable to display top holdings.")
            st.write(f"Error: {e}")
    else:
        st.write("No top holdings data available for this ETF.")

    # Performance Graph
    st.header("Performance Graph")
    today = datetime.date.today()
    start_of_year = datetime.date(today.year, 1, 1)
    start_date = st.date_input("Start Date", value=start_of_year, key="start_date")
    end_date = st.date_input("End Date", value=today, key="end_date")

    # Fetch price data for the selected ETF
    query = '''
        SELECT Date, Close
        FROM etf_prices
        WHERE symbol = ? AND Date BETWEEN ? AND ?
        ORDER BY Date ASC
    '''
    price_data = pd.read_sql_query(query, conn, params=(selected_etf, start_date, end_date))

    # Calculate Time-Weighted Return (TWR)
    twr = None
    if not price_data.empty:
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data.set_index('Date', inplace=True)

        # Calculate TWR
        start_price = price_data['Close'].iloc[0]
        end_price = price_data['Close'].iloc[-1]
        twr = ((end_price / start_price) - 1) * 100  # Convert to percentage

    # Display Performance Graph Header with TWR
    if twr is not None:
        st.markdown(f"<h2 style='text-align: center;'>Performance Graph (Time-Weighted Return: {twr:.2f}%)</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center;'>Performance Graph (No Data Available)</h2>", unsafe_allow_html=True)

    # Plot the performance graph using Plotly
    if not price_data.empty:
        price_data.reset_index(inplace=True)  # Reset index to use Date as a column
        fig = px.line(
            price_data,
            x='Date',
            y='Close',
            title=f"{selected_etf} Performance",
            labels={'Close': 'Closing Price', 'Date': 'Date'},
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Closing Price",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No price data available for the selected date range.")

    

    # Close the database connection
    conn.close()
