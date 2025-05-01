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

    # Fetch models and security sets
    @st.cache_data
    def load_models_and_security_sets():
        # Fetch all models by joining model_security_set with models
        cursor.execute('''
            SELECT DISTINCT models.name
            FROM model_security_set
            JOIN models ON model_security_set.model_id = models.id
        ''')
        models = [row[0] for row in cursor.fetchall()]

        # Fetch all security sets by joining model_security_set with security_sets
        cursor.execute('''
            SELECT DISTINCT security_sets.name
            FROM model_security_set
            JOIN security_sets ON model_security_set.security_set_id = security_sets.id
        ''')
        security_sets = [row[0] for row in cursor.fetchall()]

        return models, security_sets

    # Fetch ETFs based on filters
    @st.cache_data
    def load_filtered_etfs(selected_model, selected_security_set):
        query = '''
            SELECT etfs.symbol
            FROM etfs
            JOIN security_sets_etfs ON etfs.symbol = security_sets_etfs.etf_id
            JOIN security_sets ON security_sets_etfs.security_set_id = security_sets.id
            JOIN model_security_set ON security_sets.id = model_security_set.security_set_id
            JOIN models ON model_security_set.model_id = models.id
            WHERE 1=1
        '''
        params = []

        if selected_model != "All Models":
            query += ' AND models.name = ?'
            params.append(selected_model)

        if selected_security_set != "All Security Sets":
            query += ' AND security_sets.name = ?'
            params.append(selected_security_set)

        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    # Load models and security sets
    models, security_sets = load_models_and_security_sets()

    # Sidebar: Filters
    st.sidebar.title("Filters")

    # Model filter
    selected_model = st.sidebar.selectbox(
        "Filter by Model",
        ["All Models"] + models,
        key="model_filter"
    )

    # Security set filter
    filtered_security_sets = ["All Security Sets"]
    if selected_model != "All Models":
        # Fetch security sets associated with the selected model
        cursor.execute('''
            SELECT DISTINCT security_sets.name
            FROM model_security_set
            JOIN security_sets ON model_security_set.security_set_id = security_sets.id
            JOIN models ON model_security_set.model_id = models.id
            WHERE models.name = ?
        ''', (selected_model,))
        filtered_security_sets += [row[0] for row in cursor.fetchall()]
    else:
        # Fetch all security sets if no model is selected
        cursor.execute('SELECT DISTINCT name FROM security_sets')
        filtered_security_sets += [row[0] for row in cursor.fetchall()]

    selected_security_set = st.sidebar.selectbox(
        "Filter by Security Set",
        filtered_security_sets,
        key="security_set_filter"
    )

    # Load filtered ETFs
    myTickers = load_filtered_etfs(selected_model, selected_security_set)

    # If no filters are applied, show all ETFs
    if not myTickers:
        cursor.execute('SELECT symbol FROM etfs')
        myTickers = [row[0] for row in cursor.fetchall()]

    # Sidebar: ETF selection
    st.sidebar.title('Foguth Financial ETF Lookup Tool')
    if myTickers:
        selected_etf = st.sidebar.selectbox(
            'Select an ETF',
            myTickers,
            key='etf_selectbox'
        )
    else:
        st.sidebar.write("No ETFs match the selected filters.")
        selected_etf = None

    # Cache ETF data
    @st.cache_data
    def load_etf_data(myTickers):
        tickerdata = {}

        for etf in myTickers:
            # Fetch data for the ETF from the etf_infos table
            cursor.execute('SELECT * FROM etf_infos WHERE symbol = ?', (etf,))
            result = cursor.fetchone()
            if result:
                # Get column names dynamically
                columns = [description[0] for description in cursor.description]
                etf_info = dict(zip(columns, result))

                # Extract relevant data
                long_name = etf_info.get('longName', "No name available")
                category = etf_info.get('category', "No category available")
                fund_family = etf_info.get('fundFamily', "No fund family available")
                dividend_yield = etf_info.get('dividendYield', "No dividend yield available")
                net_expense_ratio = etf_info.get('netExpenseRatio', "No expense ratio available")
                long_business_summary = etf_info.get('longBusinessSummary', "No summary available.")

                etf_dict = {
                    'Long Name': long_name,
                    'Category': category,
                    'Fund Family': fund_family,
                    'Dividend Yield': dividend_yield,
                    'Net Expense Ratio': net_expense_ratio,
                    'Long Business Summary': long_business_summary
                }
                tickerdata[etf] = etf_dict

        return tickerdata

    # Load ETF data
    tickerdata = load_etf_data(myTickers)

    # Main content: Display selected ETF information
    if selected_etf:
        st.markdown(f"<h1 style='text-align: center;'>{selected_etf} - {tickerdata[selected_etf]['Long Name']}</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"### **Summary**")
        st.write(tickerdata[selected_etf]['Long Business Summary'])

        st.markdown(f"### **Category**")
        st.write(f"{tickerdata[selected_etf]['Category']}")

        st.markdown(f"### **Fund Manager**")
        st.write(f"{tickerdata[selected_etf]['Fund Family']}")

        st.markdown(f"### **Dividend Yield**")
        st.write(f"{tickerdata[selected_etf]['Dividend Yield']}%")

        st.markdown(f"### **Net Expense Ratio**")
        st.write(f"{tickerdata[selected_etf]['Net Expense Ratio']}%")

        # Performance Graph
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
    else:
        st.write("Please select an ETF to view details.")

    # Close the database connection
    conn.close()
