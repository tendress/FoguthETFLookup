import streamlit as st
from updateytdreturnsmodule import update_etf_ytd_returns, update_security_set_ytd_returns, update_model_ytd_returns

# Path to the SQLite database
database_path = 'foguth_etf_models.db'

# Streamlit app title
st.title("YTD Returns Updater")

# Instructions
st.write("Use the buttons below to update the YTD returns for ETFs, Security Sets, and Models.")

# Button to update ETF YTD returns
if st.button("Update ETF YTD Returns"):
    st.write("Updating ETF YTD returns...")
    etf_df = update_etf_ytd_returns(database_path)
    st.write("ETF YTD returns updated successfully!")
    st.dataframe(etf_df)

# Button to update Security Set YTD returns
if st.button("Update Security Set YTD Returns"):
    st.write("Updating Security Set YTD returns...")
    security_set_df = update_security_set_ytd_returns(database_path)
    st.write("Security Set YTD returns updated successfully!")
    st.dataframe(security_set_df)

# Button to update Model YTD returns
if st.button("Update Model YTD Returns"):
    st.write("Updating Model YTD returns...")
    model_df = update_model_ytd_returns(database_path)
    st.write("Model YTD returns updated successfully!")
    st.dataframe(model_df)