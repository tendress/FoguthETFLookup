import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime

def display_live_factsheet():
    st.markdown("""
        <style>
        @media print {
            /* Hide the Streamlit sidebar when printing */
            section[data-testid="stSidebar"] {
                display: none !important;
            }
            /* Expand the main content to full width */
            section[data-testid="stMain"] {
                width: 100vw !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    @st.cache_data
    def load_etf_weights_for_model(selected_model):
        query = '''
            SELECT 
                etfs.symbol AS ETF, 
                etfs.name AS Name,
                etfs.yield AS Yield,
                SUM(security_sets_etfs.weight * model_security_set.weight) AS ModelWeight
            FROM model_security_set
            JOIN security_sets ON model_security_set.security_set_id = security_sets.id
            JOIN security_sets_etfs ON security_sets.id = security_sets_etfs.security_set_id
            JOIN etfs ON security_sets_etfs.etf_id = etfs.id
            JOIN models ON model_security_set.model_id = models.id
            WHERE models.name = ?
            AND security_sets_etfs.endDate IS NULL
            GROUP BY etfs.symbol, etfs.name, etfs.yield
            ORDER BY ModelWeight DESC
        '''
        conn = sqlite3.connect("foguth_etf_models.db")
        df = pd.read_sql_query(query, conn, params=(selected_model,))
        conn.close()
        # Convert ModelWeight to percentage and add '%' sign
        if not df.empty:
            total_weight = df["ModelWeight"].sum()
            df["ModelWeight"] = (df["ModelWeight"] / total_weight * 100).round(2).astype(str) + '%'
            df["ModelWeightValue"] = (df["ModelWeight"].str.rstrip('%').astype(float))  # For pie chart
            df.rename(columns={"ModelWeight": "ModelWeight (%)"}, inplace=True)
        return df

    def parse_mixed_dates(val):
        try:
            # Try as Unix timestamp (int or float)
            return pd.to_datetime(float(val), unit='s')
        except (ValueError, TypeError):
            try:
                # Try as date string
                return pd.to_datetime(val)
            except Exception:
                return pd.NaT
    
    
    
    def load_models():
        conn = sqlite3.connect("foguth_etf_models.db")
        models = pd.read_sql_query("SELECT name FROM models", conn)["name"].tolist()
        conn.close()
        return models

    def get_categories_for_etfs(etf_symbols):
        if not etf_symbols:
            return pd.DataFrame(columns=["category", "Weight"])
        conn = sqlite3.connect("foguth_etf_models.db")
        placeholders = ','.join(['?'] * len(etf_symbols))
        query = f'''
            SELECT symbol, category FROM etf_infos
            WHERE symbol IN ({placeholders})
        '''
        df = pd.read_sql_query(query, conn, params=etf_symbols)
        conn.close()
        return df


    def plot_ytd_and_range_bar_chart(model_returns_df, sp500_df, selected_model, start_date, end_date):
        """
        Plots a grouped bar chart comparing:
        - YTD return of the model vs S&P 500 (^GSPC)
        - Selected date range return of the model vs S&P 500 (^GSPC)
        """
        import plotly.express as px
        import pandas as pd

        # Ensure dates are datetime
        model_returns_df["Date"] = pd.to_datetime(model_returns_df["Date"])
        sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])

        # --- YTD Calculation ---
        ytd_start = pd.Timestamp(year=pd.Timestamp.today().year, month=1, day=1)
        ytd_end = model_returns_df["Date"].max()

        # Model YTD
        model_ytd_df = model_returns_df[(model_returns_df["Date"] >= ytd_start) & (model_returns_df["Date"] <= ytd_end)]
        if not model_ytd_df.empty:
            model_ytd_return = (model_ytd_df["cum_return"].iloc[-1] - model_ytd_df["cum_return"].iloc[0]) / 100
        else:
            model_ytd_return = 0

        # S&P 500 YTD
        sp500_ytd_df = sp500_df[(sp500_df["Date"] >= ytd_start) & (sp500_df["Date"] <= ytd_end)]
        if not sp500_ytd_df.empty:
            sp500_ytd_return = (sp500_ytd_df["cum_return"].iloc[-1] - sp500_ytd_df["cum_return"].iloc[0])
        else:
            sp500_ytd_return = 0

        # --- Selected Date Range Calculation ---
        model_range_df = model_returns_df[(model_returns_df["Date"] >= pd.to_datetime(start_date)) & (model_returns_df["Date"] <= pd.to_datetime(end_date))]
        if not model_range_df.empty:
            model_range_return = (model_range_df["cum_return"].iloc[-1] - model_range_df["cum_return"].iloc[0]) / 100
        else:
            model_range_return = 0

        sp500_range_df = sp500_df[(sp500_df["Date"] >= pd.to_datetime(start_date)) & (sp500_df["Date"] <= pd.to_datetime(end_date))]
        if not sp500_range_df.empty:
            sp500_range_return = (sp500_range_df["cum_return"].iloc[-1] - sp500_range_df["cum_return"].iloc[0])
        else:
            sp500_range_return = 0

        # --- Prepare DataFrame for Grouped Bar Plot ---
        bar_data = pd.DataFrame({
            "Period": ["YTD", "YTD", "Range", "Range"],
            "Return Type": [selected_model, "S&P 500", selected_model, "S&P 500"],
            "Return (%)": [
                round(model_ytd_return * 100, 2),
                round(sp500_ytd_return * 100, 2),
                round(model_range_return * 100, 2),
                round(sp500_range_return * 100, 2)
            ]
        })

        fig = px.bar(
            bar_data,
            x="Period",
            y="Return (%)",
            color="Return Type",
            barmode="group",
            text="Return (%)",
            title=f"YTD and Selected Range Returns: {selected_model} vs S&P 500"
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        # Give extra space at the top so text is not cut off
        max_y = bar_data["Return (%)"].max()
        fig.update_layout(
            yaxis_title="Return (%)",
            margin=dict(t=60),  # Increase top margin
            yaxis_range=[None, max_y * 1.25]  # Add 25% headroom
        )
        fig.update_layout(yaxis_title="Return (%)")
        st.plotly_chart(fig, use_container_width=True)

    def calculate_model_time_weighted_return(selected_model):
        """
        Calculate the model's time-weighted return by aggregating the weighted returns
        of its security sets using model_security_set.weight and security_set_prices.percentChange.
        Returns a DataFrame with columns: Date, cum_return.
        """
        conn = sqlite3.connect("foguth_etf_models.db")

        # Get security sets and their weights for the selected model
        query_sets = """
            SELECT ms.security_set_id, ms.weight
            FROM model_security_set ms
            JOIN models m ON ms.model_id = m.id
            WHERE m.name = ?
        """
        sets_df = pd.read_sql_query(query_sets, conn, params=(selected_model,))

        if sets_df.empty:
            conn.close()
            return pd.DataFrame(columns=["Date", "cum_return"])

        # Get all security set prices for these sets
        set_ids = sets_df["security_set_id"].tolist()
        placeholders = ','.join(['?'] * len(set_ids))
        query_prices = f"""
            SELECT security_set_id, Date, percentChange
            FROM security_set_prices
            WHERE security_set_id IN ({placeholders})
            ORDER BY Date ASC
        """
        prices_df = pd.read_sql_query(query_prices, conn, params=set_ids)
        conn.close()

        if prices_df.empty:
            return pd.DataFrame(columns=["Date", "cum_return"])

        # Merge weights into prices
        merged = prices_df.merge(sets_df, left_on="security_set_id", right_on="security_set_id", how="left")
        merged["weighted_return"] = merged["percentChange"] * merged["weight"]

        # Group by Date and sum weighted returns to get model return for each date
        model_returns = merged.groupby("Date")["weighted_return"].sum().reset_index()
        model_returns = model_returns.rename(columns={"weighted_return": "model_return"})
        model_returns["Date"] = pd.to_datetime(model_returns["Date"])
        model_returns = model_returns.sort_values("Date")

        # Calculate cumulative time-weighted return (assuming percentChange is daily %)
        model_returns["cum_return"] = (1 + model_returns["model_return"] / 100).cumprod() - 1
        model_returns["cum_return"] = model_returns["cum_return"] * 100  # as percentage

        return model_returns[["Date", "cum_return"]]

    models = load_models()
    selected_model = st.sidebar.selectbox(
        "Filter by Model",
        ["All Models"] + models,
        key="model_filter"
    )

    fscol1, fscol2 = st.columns([.3,.7])
    with fscol1:
        st.image("assets/fwmlogo.png", width=200, )
    with fscol2:
        st.title("Foguth Wealth Management ETP Fact Sheet")

    # add a line below the title
    st.markdown("---")




    if selected_model != "All Models":
        etf_df = load_etf_weights_for_model(selected_model)
        if not etf_df.empty:
            # Display the selected Model name
            # style the header
            st.markdown("<h2 style='color: #ffffff; background-color:#336699; padding-left:4pt;'>Selected Model: {}</h2>".format(selected_model), unsafe_allow_html=True)
            
            # Prepare top 10 securities DataFrame (drop index column)
            top10_df = etf_df[["ETF", "Name", "ModelWeight (%)"]].reset_index(drop=True)

            # Pie chart of ETF categories
            etf_symbols = etf_df["ETF"].tolist()
            category_df = get_categories_for_etfs(etf_symbols)
            merged = pd.merge(etf_df, category_df, left_on="ETF", right_on="symbol", how="left")
            merged = merged.dropna(subset=["category"])
            merged["category"] = merged["category"].str.strip()
            merged["Weight"] = merged["ModelWeight (%)"].str.rstrip('%').astype(float)
            pie_data = merged.groupby("category")["Weight"].sum().reset_index()
            pie_data = pie_data[pie_data["category"].notnull() & (pie_data["category"] != "")]

            # Display side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Security Target Weights")
                st.dataframe(top10_df)
            with col2:
                st.subheader("Asset Allocation")
                if not pie_data.empty:
                    fig = px.pie(pie_data, names="category", values="Weight")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No category data available for the selected model.")

            # --- Model and S&P 500 Growth Chart with Date Picker ---
            model_returns_df = calculate_model_time_weighted_return(selected_model)
            if not model_returns_df.empty:
                st.markdown("<h4 style='color: #ffffff; background-color:#336699;  padding-left:4pt;'>Growth of $1,000,000 for {}</h2>".format(selected_model), unsafe_allow_html=True)
                initial_investment = 1_000_000
                model_returns_df["growth"] = (initial_investment * (1 + model_returns_df["cum_return"] / 100)).round(0)

                # --- Get ^GSPC (S&P 500) returns and calculate growth ---
                conn = sqlite3.connect("foguth_etf_models.db")
                sp500_df = pd.read_sql_query(
                    "SELECT Date, Close FROM etf_prices WHERE symbol = '^GSPC' ORDER BY Date ASC",
                    conn
                )
                conn.close()
                if not sp500_df.empty:
                    sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])  # Convert to datetime
                    sp500_df["Date"] = sp500_df["Date"].apply(parse_mixed_dates)  # Handle mixed date formats
                    sp500_df = sp500_df[sp500_df["Date"].isin(model_returns_df["Date"])]
                    sp500_df = sp500_df.sort_values("Date")
                    sp500_df["pct_change"] = sp500_df["Close"].pct_change().fillna(0)
                    sp500_df["cum_return"] = (1 + sp500_df["pct_change"]).cumprod() - 1
                    sp500_df["growth"] = (initial_investment * (1 + sp500_df["cum_return"])).round(0)

                    # --- Date Picker and Filtering ---
                    min_date = datetime.datetime(2025, 1, 1)  # Set minimum date to Jan 1, 2025
                    max_date = datetime.datetime.now().date()  # Set maximum date to today

                    
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_date = st.date_input(
                            "Start Date",
                            value=min_date,
                            # set min value to Jan 1, 2025
                            min_value=min_date,
                            max_value=max_date,
                            key="start_date_picker"
                        )
                    with col_end:
                        end_date = st.date_input(
                            "End Date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="end_date_picker"
                        )

                    # Filter both model_returns_df and sp500_df by the selected date range
                    mask = (model_returns_df["Date"] >= pd.to_datetime(start_date)) & (model_returns_df["Date"] <= pd.to_datetime(end_date))
                    filtered_model_returns_df = model_returns_df.loc[mask].copy()

                    mask_sp = (sp500_df["Date"] >= pd.to_datetime(start_date)) & (sp500_df["Date"] <= pd.to_datetime(end_date))
                    filtered_sp500_df = sp500_df.loc[mask_sp].copy()

                    # Rebase both series to $1,000,000 at the selected start date
                    if not filtered_model_returns_df.empty and not filtered_sp500_df.empty:
                        model_start = filtered_model_returns_df["growth"].iloc[0]
                        filtered_model_returns_df["growth_rebased"] = filtered_model_returns_df["growth"] / model_start * 1_000_000

                        sp500_start = filtered_sp500_df["growth"].iloc[0]
                        filtered_sp500_df["growth_rebased"] = filtered_sp500_df["growth"] / sp500_start * 1_000_000

                        # Merge for plotting
                        plot_df = pd.merge(
                            filtered_model_returns_df[["Date", "growth_rebased"]].rename(columns={"growth_rebased": selected_model}),
                            filtered_sp500_df[["Date", "growth_rebased"]].rename(columns={"growth_rebased": "S&P 500"}),
                            on="Date",
                            how="inner"
                        )

                        # --- Plot line and bar chart side by side ---
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            fig = px.line(
                                plot_df,
                                x="Date",
                                y=[selected_model, "S&P 500"],
                                title=f"Growth of $1,000,000: {selected_model} vs S&P 500",
                                labels={"value": "Portfolio Value ($)", "variable": "Investment"}
                            )
                            # round y-axis values to 0 decimal places
                            fig.update_layout(
                                yaxis_title="Portfolio Value ($)",
                                xaxis_title="Date",
                                legend_title_text="Investment",
                                legend=dict(x=1, y=1, traceorder="normal"),
                                xaxis=dict(tickformat="%Y-%m-%d"),
                                yaxis=dict(tickformat="$,.0f"),  # Format y-axis as currency with no decimal places
                                title=dict(text=f"Growth of $1,000,000: {selected_model} vs S&P 500"),
                                font=dict(size=12),
                                margin=dict(t=60)  # Increase top margin for title
                            )
                            
                            # round the tick values on y-axis to 0 decimal places
                            fig.update_yaxes(tickformat="$,.0f")
                            fig.update_xaxes(tickformat="%Y-%m-%d")
                            st.plotly_chart(fig, use_container_width=True)
                        with chart_col2:
                            plot_ytd_and_range_bar_chart(
                                model_returns_df,
                                sp500_df,
                                selected_model,
                                start_date,
                                end_date
                            )
                        
                        st.markdown("---")
                        st.markdown("<h2 style='color: #ffffff; background-color:#336699; padding-left:4pt;'>Disclosures</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='padding-top:10pt;'>Past performance is not indicative of future performance. Principal value and investment return will fluctuate. There are no implied guarantees or assurances that the target returns will be achieved or objectives will be met. Future returns may differ significantly from past returns due to many different factors. Investments involve risk and the possibility of loss of principal. The values and performance numbers represented in this report are inclusive of management fees. The values used in this report were obtained from sources believed to be reliable. Performance numbers were calculated by Foguth Wealth Management using publicly available data. Please consult your custodial statements for an official record of value.</p>", unsafe_allow_html=True)
                        st.markdown("<p style='padding-top:10pt;'>Hypothetical, back-tested performance results have inherent limitations, including the following: (1) the results do not reflect the results of actual trading, but were achieved by means of retroactive application, which may have been designed with the benefit of hindsight; (2) back-tested performance may not reflect the impact that any material market or economic factors might have had on Foguth Wealth Management’s decision making process had the strategy been used during the stated period to actually manage client assets; and (3) for various reasons (including the reasons indicated above), clients may have experienced investment results during the corresponding time periods that we materially different from those portrayed. Back-tested returns do not represent actual returns and should not be interpreted as an indication of such</p>", unsafe_allow_html=True)
                        st.markdown("<p style='padding-top:10pt;'>The assumptions and projections displayed are estimates, hypothetical in nature, and meant to serve solely as a guideline. The results and analysis are not guarantees of future results because they are derived from mathematical modeling techniques of the economic and financial markets that may or may not reflect actual conditions and events. Accordingly, you should not rely solely on the information contained in these materials in making any investment decision.</p>", unsafe_allow_html=True)
                        st.markdown("<p style='padding-top:10pt;'>Performance numbers are compared to an index for benchmarking purposes. The index chosen reflects all applicable dividends reinvested. The index results reflect fees and expenses, and you typically cannot invest in an index.</p>", unsafe_allow_html=True)
                        st.markdown("<p style='padding-top:10pt;'>The S&P 500 Index is a market-cap weighted index composed of the common stock of 500 leading companies in leading industries of the U.S. economy. The benchmark presented represent an unmanaged portfolio whose charac¬teristics differ from the strategies; however, they tend to represent the investment environment existing during the time periods shown. The benchmark cannot be invested in directly. The returns of the benchmark do not include any transaction costs, management fees or other costs. The holdings of the client portfolios in the strategies may differ significantly from the securities that comprise the benchmark shown. The benchmark has been selected to represent what Foguth Wealth Management believes is an appropriate benchmark with which to compare the performance of the strategies</p>", unsafe_allow_html=True)

                        st.markdown("---")
                        
                        imcol1, imcol2, imcol3 = st.columns([.3,.4,.3])
                        with imcol1:
                            st.markdown("---")
                            st.markdown("<p style='text-align: center; font-size: 12px;'></p>", unsafe_allow_html=True)
                        with imcol2:
                            st.image("assets/fwmlogo.png", width=200, use_container_width=True)
                        with imcol3:
                            st.markdown("---")
                            st.markdown("<p style='text-align: center; font-size: 12px;'>All rights reserved</p>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("<h2 style='text-align:center;'>Contact Us</h2>", unsafe_allow_html=True)
                        st.markdown("<h3 style='text-align:center;'><a>www.foguthfinancial.com</a> | (844)-4-FOGUTH</h3>", unsafe_allow_html=True)
                        
                    elif not filtered_model_returns_df.empty:
                        st.write("No S&P 500 (^GSPC) data found for comparison in the selected date range.")
                    else:
                        st.write("No model return data found for the selected date range.")
                else:
                    st.write("No S&P 500 (^GSPC) data found for comparison.")
            else:
                st.write("No calculated model return data found for this model.")
        else:
            st.write("No ETFs found for the selected model.")
    else:
        st.write("Select a model to view its Fact Sheet.")