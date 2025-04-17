### Foguth Financial ETF Lookup Tool ###
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import openpyxl
import matplotlib


def etflookup(etf):
    # Get the ETF data
    etf_data = yf.Ticker(etf)
    #etf_info = etf_data.info
    #etf_balance_sheet = etf_data.balance_sheet
    etf_trades = etf_data.get_funds_data()
    
    #return etf_info
    return etf_trades

@st.cache_data
def load_etf_data(etf_tickers):
    tickerdata = {}
    etf_corr = pd.DataFrame()

    for etf in etf_tickers:
        etf_info = etflookup(etf)
        # Get the data for the individual ETFs
        top_holdings = etf_info.top_holdings
        top_holdings['Holding Percent'] = top_holdings['Holding Percent'] * 100
        # add a percent symbol after the number
        asset_classes = etf_info.asset_classes
        sector_weightings = etf_info.sector_weightings
        fund_overview = etf_info.fund_overview
        fund_operations = etf_info.fund_operations
        fund_description = etf_info.description
        # store the data in a dictionary with the ticker as the key
        etf_data = {'Top Holdings': top_holdings, 'Asset Classes': asset_classes, 'Sector Weightings': sector_weightings, 'Fund Overview': fund_overview, 'Fund Operations': fund_operations, 'Fund Description': fund_description}
        tickerdata[etf] = etf_data

        etf_data = yf.Ticker(etf)
        etf_history = etf_data.history(period='max')
        etf_corr[etf] = etf_history['Close']
    
    etf_corr = etf_corr.pct_change().corr()
    return tickerdata, etf_corr

#etf_tickers = pd.read_excel(r'tickers1.xlsx')
#etf_tickers = etf_tickers['Ticker'].tolist()
etf_tickers = ['QQQM', 'SPY', 'DIVB', 'MOAT', 'AGG', 'SPYG', 'TLT', 'LQD', 'SPSM', 'UTEN', 'UFIV', 'VTI', 'PHYL', 'FLBL', 'BIL', 'SPYV', 'FJAN', 'FJUN', 'DJUL', 'DOCT', 'XLC', 'XLE', 'XLF', 'XLV', 'XLY', 'XLK', 'OMFL', 'AIQ', 'SMH', 'IBLC', 'BOTZ', 'GRID', 'JEPI', 'SCHD', 'QYLD', 'DIVO', 'FNDX', 'IUS', 'SDIV', 'SPHD', 'KNG', 'PAAA', 'NCLO']


# Load ETF data and correlation matrix
tickerdata, etf_corr = load_etf_data(etf_tickers)
# Initialize session state
if 'selected_etf' not in st.session_state:
    st.session_state.selected_etf = etf_tickers[0]

# Create a Streamlit sidebar that lets the user select an ETF
st.sidebar.title('Foguth Financial ETF Lookup Tool')
st.sidebar.write("https://foguthetflookup.streamlit.app/")
selected_etf = st.sidebar.selectbox(
    'Select an ETF',
    etf_tickers
)

# read the json file foguthmodels.json
foguthmodels = pd.read_json('foguthmodels.json')
foguthmodels = foguthmodels.to_dict()

# Create a Streamlit sidebar that lets the user select a model
selected_model = st.sidebar.selectbox(
    'Select a Model',
    foguthmodels.keys()
)

# After the user selects the model, display the weights for the selected model in the sidebar
# and do not display tickers that have a weight of 0
# Display the model weights in the sidebar
# sort the dictionary by value
sorted_weights = sorted(foguthmodels[selected_model].items(), key=lambda x: x[1], reverse=True)
for ticker, weight in sorted_weights:
    if weight > 0:
        st.sidebar.write(f'{ticker}: {weight}')
        



# Update the selected ETF in session state
st.session_state.selected_etf = selected_etf

# Display the data for the selected ETF
# Streamlit headline
st.title('ETF Information')
# Style a Streamlit header for the selected ETF
st.header(selected_etf, divider=True)
st.write(tickerdata[selected_etf]['Fund Description'])
st.write(tickerdata[selected_etf]['Fund Overview'])
# Streamlit header
st.header('Top Holdings', divider=True)
st.write(tickerdata[selected_etf]['Top Holdings'])
st.header('Asset Classes', divider=True)
st.write(tickerdata[selected_etf]['Asset Classes'])
st.header('Sector Weightings', divider=True)
st.write(tickerdata[selected_etf]['Sector Weightings'])
st.header('Fund Operations', divider=True)
st.write(tickerdata[selected_etf]['Fund Operations'])

#display a graph of performance history
etf_data = yf.Ticker(selected_etf)
etf_history = etf_data.history(period='max')
st.header('Performance History', divider=True)
st.line_chart(etf_history['Close'])


st.header('Correlation Matrix', divider=True)
st.write(etf_corr.style.background_gradient(cmap='YlGn'))
