import sqlite3
import pandas as pd
import streamlit as st

# Streamlit app title
st.title("Models Table Viewer")

# Connect to SQLite database
def load_models_table():
    try:
        # Connect to the database
        conn = sqlite3.connect('foguth_etf_models.db')
        
        # Load the models table into a DataFrame
        query = "SELECT name AS Name, YTDPriceReturn AS YTDReturn, YTDPriceReturnDate AS AsOf, yield AS Yield FROM models"
        models_df = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        return models_df
    except Exception as e:
        st.error(f"Error loading models table: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error
    
    

def load_security_sets_table():
    try:
        # Connect to the database
        conn = sqlite3.connect('foguth_etf_models.db')
        
        # Load the security_sets table into a DataFrame
        query = "SELECT name AS Name, YTDPriceReturn AS YTDReturn, YTDPriceReturnDate AS AsOf, yield as Yield FROM security_sets"
        security_sets_df = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        return security_sets_df
    except Exception as e:
        st.error(f"Error loading Security Sets table: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Load the models, security_sets table
models_df = load_models_table()
security_sets_df = load_security_sets_table()

# Display the models table in the Streamlit app and remove the id column and sort by YTDPriceReturn
if not models_df.empty:
    # Remove the 'id' column if it exists
    if 'id' in models_df.columns:
        models_df = models_df.drop(columns=['id'])
    
    # Sort by YTDPriceReturn in descending order
    if 'YTDPriceReturn' in models_df.columns:
        models_df = models_df.sort_values(by='YTDPriceReturn', ascending=False).reset_index(drop=True)
    
    # Display the DataFrame in the Streamlit app
    st.dataframe(models_df, use_container_width=True, height=500, hide_index=True)
else:
    st.warning("No data available in the models table.")
    

# Display the security_sets table in the Streamlit app and remove the id column and sort by YTDPriceReturn
if not security_sets_df.empty:
    # Remove the 'id' column if it exists
    if 'id' in security_sets_df.columns:
        security_sets_df = security_sets_df.drop(columns=['id'])
    
    # Sort by YTDPriceReturn in descending order
    if 'YTDPriceReturn' in security_sets_df.columns:
        security_sets_df = security_sets_df.sort_values(by='YTDPriceReturn', ascending=False).reset_index(drop=True)
    
    # Display the DataFrame in the Streamlit app
    st.dataframe(security_sets_df, use_container_width=True, height=500, hide_index=True)
    

        