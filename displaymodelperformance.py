import sqlite3
import pandas as pd
import streamlit as st
import yfinance as yf
from updateytdreturnsmodule import update_etf_ytd_returns, update_security_set_ytd_returns, update_model_ytd_returns

def display_model_performance():
    st.title("Model Performance")
    st.write("This page displays model performance metrics.")

    database_path = 'foguth_etf_models.db'

    # Run update functions first
    st.write("Updating ETF YTD returns...")
    etf_df = update_etf_ytd_returns(database_path)
    #st.write("ETF YTD returns updated successfully!")

    #st.write("Updating Security Set YTD returns...")
    security_set_df = update_security_set_ytd_returns(database_path)
    #
    #st.write("Updating Model YTD returns...")
    model_df = update_model_ytd_returns(database_path)
    st.write("Model YTD returns updated successfully!")

    # Load the models table
    def load_models_table():
        try:
            # Connect to the database
            conn = sqlite3.connect(database_path)
            
            # Load the models table into a DataFrame
            query = "SELECT name AS Name, YTDPriceReturn AS YTDReturn, YTDPriceReturnDate AS AsOf, yield AS AnnualYield, ExpenseRatio FROM models"
            models_df = pd.read_sql_query(query, conn)
            
            # Close the connection
            conn.close()
            
            return models_df
        except Exception as e:
            st.error(f"Error loading models table: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if there's an error

    # Load the security_sets table
    def load_security_sets_table():
        try:
            # Connect to the database
            conn = sqlite3.connect(database_path)
            
            # Load the security_sets table into a DataFrame
            query = "SELECT name AS Name, YTDPriceReturn AS YTDReturn, YTDPriceReturnDate AS AsOf, yield as Yield FROM security_sets"
            security_sets_df = pd.read_sql_query(query, conn)
            
            # Close the connection
            conn.close()
            
            return security_sets_df
        except Exception as e:
            st.error(f"Error loading Security Sets table: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if there's an error

    # Load the models and security_sets tables
    models_df = load_models_table()
    security_sets_df = load_security_sets_table()

    # Display the models table
    st.header("Year-To-Date Model Performance")
    if not models_df.empty:
        # Sort by YTDPriceReturn in descending order
        if 'YTDReturn' in models_df.columns:
            models_df = models_df.sort_values(by='YTDReturn', ascending=False).reset_index(drop=True)
        
        # Format the Yield and ExpenseRatio columns
        if 'ExpenseRatio' in models_df.columns:
            models_df['ExpenseRatio'] = models_df['ExpenseRatio'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        
        # Display the DataFrame
        st.dataframe(models_df, use_container_width=True, height=500, hide_index=True)
    else:
        st.warning("No data available in the models table.")

    # Display the security_sets table
    st.header("Year-To-Date Security Sets Performance")
    if not security_sets_df.empty:
        # Sort by YTDPriceReturn in descending order
        if 'YTDReturn' in security_sets_df.columns:
            security_sets_df = security_sets_df.sort_values(by='YTDReturn', ascending=False).reset_index(drop=True)
        
        # Display the DataFrame
        st.dataframe(security_sets_df, use_container_width=True, height=500, hide_index=True)
    else:
        st.warning("No data available in the security sets table.")

    # Sidebar for benchmark YTD performance
    def get_ytd_price_return(ticker):
        try:
            # Fetch the ticker data from yfinance
            etf = yf.Ticker(ticker)
            ytd_data = etf.history(period='ytd')
            
            if not ytd_data.empty:
                start_price = ytd_data['Close'].iloc[0]  # Price on the first trading day of the year
                current_price = ytd_data['Close'].iloc[-1]  # Latest price
                ytd_price_return = ((current_price - start_price) / start_price) * 100 if start_price else None
                return ytd_price_return
            else:
                return None
        except Exception as e:
            st.error(f"Error fetching YTD price return for {ticker}: {e}")
            return None

    # Display benchmark YTD performance in the sidebar
    st.sidebar.title("Benchmarks")
    spy_ytd_return = get_ytd_price_return('SPY')
    qqqm_ytd_return = get_ytd_price_return('QQQM')
    dia_ytd_return = get_ytd_price_return('DIA')
    st.sidebar.write(f"S&P 500 YTD: {spy_ytd_return:.2f}%" if spy_ytd_return is not None else "SPY data not available")   
    st.sidebar.write(f"Nasdaq YTD: {qqqm_ytd_return:.2f}%" if qqqm_ytd_return is not None else "QQQM data not available")
    st.sidebar.write(f"Dow Jones YTD: {dia_ytd_return:.2f}%" if dia_ytd_return is not None else "DIA data not available")
