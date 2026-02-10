### Foguth Financial ETF Lookup Tool ###
import datetime
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


def etf_lookup():
    st.title("Model & ETF Lookup")

    database_path = "foguth_etf_models.db"

    @st.cache_data(ttl=30)
    def load_all_etfs_with_names():
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, name FROM etfs")
        result = cursor.fetchall()
        conn.close()
        return result

    @st.cache_data(ttl=30)
    def load_models_and_security_sets():
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT models.name
            FROM model_security_set
            JOIN models ON model_security_set.model_id = models.id
            """
        )
        models = [row[0] for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT DISTINCT security_sets.name
            FROM security_sets
            """
        )
        security_sets = [row[0] for row in cursor.fetchall()]
        conn.close()

        return models, security_sets

    @st.cache_data(ttl=30)
    def load_security_sets_and_etfs_for_model(selected_model):
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        query = """
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
        """
        cursor.execute(query, (selected_model,))
        results = cursor.fetchall()
        conn.close()
        return pd.DataFrame(results, columns=["Security Set", "Security Set Weight", "ETF", "ETF Weight"])

    all_etfs_with_names = load_all_etfs_with_names()
    etf_options = ["Select an ETF"] + [f"{symbol} - {name}" for symbol, name in all_etfs_with_names]

    models, security_sets = load_models_and_security_sets()

    st.sidebar.title("Filters")
    selected_model = st.sidebar.selectbox(
        "Filter by Model",
        ["All Models"] + models,
        key="model_filter",
    )

    selected_etf_option = st.sidebar.selectbox(
        "Select an ETF",
        etf_options,
        key="etf_selectbox",
        index=0,
    )

    selected_symbol = None
    if selected_etf_option != "Select an ETF":
        selected_symbol = selected_etf_option.split(" - ")[0]

    if selected_model != "All Models":
        st.sidebar.title(f"Model: {selected_model}")
        security_sets_and_etfs = load_security_sets_and_etfs_for_model(selected_model)

        if not security_sets_and_etfs.empty:
            for security_set in security_sets_and_etfs["Security Set"].unique():
                security_set_weight = security_sets_and_etfs[
                    security_sets_and_etfs["Security Set"] == security_set
                ]["Security Set Weight"].iloc[0]
                st.sidebar.subheader(f"{security_set} ({security_set_weight * 100}%)")

                etfs_in_set = security_sets_and_etfs[
                    security_sets_and_etfs["Security Set"] == security_set
                ][["ETF", "ETF Weight"]]
                for _, row in etfs_in_set.iterrows():
                    st.sidebar.write(f"- {row['ETF']} ({row['ETF Weight'] * 100}%)")
        else:
            st.sidebar.write("No security sets or ETFs found for the selected model.")
    else:
        st.sidebar.write("Select a model to view its associated security sets and ETFs.")

    if selected_model != "All Models":
        st.header(f"Model: {selected_model}")
    else:
        st.header("Pick a Model to see its Strategies and ETFs")

    if selected_model != "All Models":
        security_sets_and_etfs = load_security_sets_and_etfs_for_model(selected_model)
        if not security_sets_and_etfs.empty:
            st.write(security_sets_and_etfs)
        else:
            st.write("No security sets or ETFs found for the selected model.")

    st.header("Details for Selected ETF")
    if selected_symbol:
        st.subheader(f"{selected_symbol}")
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM etf_infos WHERE symbol = ?", (selected_symbol,))
        result = cursor.fetchone()
        etf_info = {}
        if result:
            columns = [description[0] for description in cursor.description]
            etf_info = dict(zip(columns, result))

        if etf_info:
            st.markdown(f"### **{selected_symbol} - {etf_info.get('longName', 'No name available')}**")
            st.write(f"**Category:** {etf_info.get('category', 'No category available')}")
            st.write(f"**Fund Manager:** {etf_info.get('fundFamily', 'No fund family available')}")
            st.write(f"**Dividend Yield:** {etf_info.get('dividendYield', 'No dividend yield available')}%")
            st.write(f"**Net Expense Ratio:** {etf_info.get('netExpenseRatio', 'No expense ratio available')}%")
            st.write(f"**Summary:** {etf_info.get('longBusinessSummary', 'No summary available.')}")
        else:
            st.write("No details available.")

        st.header("Top 10 Holdings")
        if etf_info.get("topHoldings"):
            try:
                top_holdings = pd.read_json(etf_info["topHoldings"])
                top_holdings.index = range(1, len(top_holdings) + 1)
                top_holdings["Holding Percent"] = top_holdings["Holding Percent"].apply(
                    lambda x: f"{x:.2f}"
                )
                st.write(top_holdings)
            except Exception as e:
                st.write(" ")
                st.write(f"Error: {e}")
        else:
            st.write(" ")

        st.header("Performance Graph")
        today = datetime.date.today()
        start_of_year = datetime.date(today.year, 1, 1)
        start_date = st.date_input("Start Date", value=start_of_year, key="start_date")
        end_date = st.date_input("End Date", value=today, key="end_date")

        query = """
            SELECT Date, Close
            FROM etf_prices
            WHERE symbol = ? AND Date BETWEEN ? AND ?
            ORDER BY Date ASC
        """
        price_data = pd.read_sql_query(query, conn, params=(selected_symbol, start_date, end_date))

        twr = None
        if not price_data.empty:
            price_data["Date"] = pd.to_datetime(price_data["Date"])
            price_data.set_index("Date", inplace=True)
            start_price = price_data["Close"].iloc[0]
            end_price = price_data["Close"].iloc[-1]
            twr = ((end_price / start_price) - 1) * 100

        if twr is not None:
            st.markdown(
                f"<h2 style='text-align: center;'>Performance Graph (Time-Weighted Return: {twr:.2f}%)</h2>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<h2 style='text-align: center;'>Performance Graph (No Data Available)</h2>",
                unsafe_allow_html=True,
            )

        if not price_data.empty:
            price_data.reset_index(inplace=True)
            fig = px.line(
                price_data,
                x="Date",
                y="Close",
                title=f"Price Performance of {selected_symbol}",
                labels={"Close": "Closing Price", "Date": "Date"},
                template="plotly_white",
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Closing Price",
                hovermode="x unified",
            )
            try:
                st.plotly_chart(fig, use_container_width=True)
            except TypeError:
                st.plotly_chart(fig)
        else:
            st.write(" ")

        conn.close()
    else:
        st.write("Select an ETF to view details, holdings, and performance.")
